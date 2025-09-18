import os
import json
from pathlib import Path

from fiftyone.core.dataset import Dataset
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_processing_utils import BaseImageProcessor
from transformers.models.clip import CLIPImageProcessorFast, CLIPModel
from tqdm import tqdm

from .utils import CACHE_DIR, get_padded_bbox_crop, timeit


EMBEDDINGS_CACHE_DIR = CACHE_DIR / "embeddings"
EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class Embedder:
    def __init__(self, embedder_name: str):
        self.embedder_name = embedder_name
        self.cache_path = EMBEDDINGS_CACHE_DIR / self.embedder_name
        self.cache_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_model_class_mapping():
        return {
            "clip": CLIPModelEmbedder,
            "clip-large": CLIPLargeModelEmbedder,
            "dinov2": DinoV2ModelEmbedder,
        }

    @staticmethod
    def from_name(embedder_name: str) -> "Embedder":
        embedder_name = embedder_name.lower()
        embedder_class = Embedder.get_model_class_mapping()[embedder_name]
        if embedder_class:
            return embedder_class(embedder_name)
        else:
            raise ValueError(f"Unknown embedder name: {embedder_name}")

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError("Subclasses must define embedding_dim")

    @timeit
    def embed_detections(
        self,
        samples,
        callback: callable = None,
    ) -> dict:
        """Embed detections from FiftyOne samples.

        Args:
            samples: FiftyOne view or dataset samples
            callback: Optional progress callback

        Returns:
            Dict mapping detection IDs to embeddings
        """
        raise NotImplementedError("Subclasses must implement embed_detections()")


class ModelEmbedder(Embedder):
    def __init__(self, embedder_name: str):
        if type(self) == ModelEmbedder:
            raise TypeError(
                "ModelEmbedder is an abstract class and cannot be instantiated directly."
            )
        super().__init__(embedder_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.get_model().to(self.device)
        self.processor = self.get_processor()

    @staticmethod
    def get_embedder(embedder_name: str) -> "ModelEmbedder":
        """Factory method to get the appropriate embedder."""
        return Embedder.from_name(embedder_name)

    @property
    def batch_size(self) -> int:
        raise NotImplementedError("Subclasses must define batch_size")

    def get_model(self) -> torch.nn.Module:
        raise NotImplementedError("Subclasses must implement get_model()")

    def get_processor(self) -> BaseImageProcessor:
        raise NotImplementedError("Subclasses must implement get_processor()")

    def _embed_batch(self, images: list[Image.Image]) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement _embed_batch()")

    @timeit
    def embed_detections(
        self,
        samples: Dataset,
        callback: callable = None,
    ) -> dict:
        """Embed detections from FiftyOne samples using the model.

        Args:
            samples: FiftyOne view or dataset samples
            field_name: Name of the detections field (e.g., 'detections', 'ground_truth')
            callback: Optional progress callback(progress, message)

        Returns:
            Dict mapping sample_id -> detection_idx -> embedding
        """
        import fiftyone as fo

        # Add the embedding field to the dataset schema so it persists
        try:
            samples.add_sample_field("detections.detections.embedding", fo.ListField)
        except ValueError:
            # Field already exists
            pass

        # Collect all detections with their metadata
        all_detections = []
        for sample in samples:
            if not hasattr(sample, "detections") or sample.detections is None:
                continue

            image_path = sample.filepath
            for idx, detection in enumerate(sample.detections.detections):
                all_detections.append(
                    {
                        "sample_id": str(sample.id),
                        "detection_idx": idx,
                        "image_path": image_path,
                        "bbox": detection.bounding_box,  # [x, y, width, height] normalized
                        "label": detection.label,
                        "detection": detection,
                        "sample": sample,  # Store reference to actual sample object
                    }
                )

        if not all_detections:
            return {}

        # Process in batches
        num_batches = (len(all_detections) + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Computing embeddings"):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, len(all_detections))
            batch_crops = []
            batch_metadata = []

            for i in range(batch_start, batch_end):
                det = all_detections[i]
                try:
                    img = Image.open(det["image_path"])

                    # Convert normalized bbox to pixel coordinates
                    img_width, img_height = img.size
                    x_norm, y_norm, w_norm, h_norm = det["bbox"]
                    x = int(x_norm * img_width)
                    y = int(y_norm * img_height)
                    w = int(w_norm * img_width)
                    h = int(h_norm * img_height)

                    # Crop the detection
                    crop = img.crop((x, y, x + w, y + h))

                    # Ensure minimum size for the model
                    if crop.size[0] < 32 or crop.size[1] < 32:
                        crop = crop.resize(
                            (max(32, crop.size[0]), max(32, crop.size[1]))
                        )

                    batch_crops.append(crop)
                    batch_metadata.append(det)

                except Exception as e:
                    print(f"Error processing detection {i}: {e}")
                    continue

            if not batch_crops:
                continue

            # Compute embeddings for batch
            batch_embeds = self._embed_batch(batch_crops)

            # Store embedding in the detection object and save samples
            saved_samples = set()
            for metadata, embedding in zip(batch_metadata, batch_embeds):
                # Set the embedding on the detection object
                metadata["detection"].embedding = embedding.cpu().numpy().tolist()

                # Save the sample (avoid duplicate saves)
                sample = metadata["sample"]
                if sample.id not in saved_samples:
                    sample.save()
                    saved_samples.add(sample.id)

            # Progress callback
            if callback:
                progress = batch_end / len(all_detections)
                callback(
                    progress, f"Processed {batch_end}/{len(all_detections)} detections"
                )

        return len(all_detections)


class CLIPModelEmbedder(ModelEmbedder):
    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def batch_size(self) -> int:
        return 64

    def _embed_batch(self, images: list[Image.Image]) -> torch.Tensor:
        processed = self.processor(images, return_tensors="pt").to(self.device)
        for k in processed:
            processed[k] = processed[k].to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**processed)
        return image_features

    def get_model(self) -> torch.nn.Module:
        return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def get_processor(self) -> BaseImageProcessor:
        return CLIPImageProcessorFast.from_pretrained("openai/clip-vit-base-patch32")


class CLIPLargeModelEmbedder(CLIPModelEmbedder):
    @property
    def embedding_dim(self) -> int:
        return 768

    @property
    def batch_size(self) -> int:
        return 64

    def get_model(self) -> torch.nn.Module:
        return CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    def get_processor(self) -> BaseImageProcessor:
        return CLIPImageProcessorFast.from_pretrained("openai/clip-vit-large-patch14")


class DinoV2ModelEmbedder(ModelEmbedder):
    @property
    def embedding_dim(self) -> int:
        return 768

    @property
    def batch_size(self) -> int:
        return 64

    def _embed_batch(self, images: list[Image.Image]) -> torch.Tensor:
        processed = self.processor(images, return_tensors="pt").to(self.device)
        for k in processed:
            processed[k] = processed[k].to(self.device)
        with torch.no_grad():
            image_features = self.model(**processed).last_hidden_state[:, 0]
        return image_features

    def get_model(self) -> torch.nn.Module:
        return AutoModel.from_pretrained("facebook/dinov2-base")

    def get_processor(self) -> BaseImageProcessor:
        return AutoImageProcessor.from_pretrained("facebook/dinov2-base")
