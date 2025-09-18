import time

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators.executor import ExecutionContext


def update_progress(
    ctx: ExecutionContext,
    operation_name: str,
    progress: float,
    label: str,
):
    """Simple utility to update progress of a long-running operation.

    Args:
        ctx: Execution context
        operation_name: Name of the operation (e.g., "embedding")
        progress: Progress value between 0 and 1
        label: Descriptive label for the current progress state
        running: Whether the operation is still running
    """
    progress_key = f"{operation_name}_progress"
    ctx.dataset.reload()
    ctx.dataset.info[progress_key] = {
        "progress": progress,
        "label": label,
    }
    ctx.dataset.save()


class FetchAnnotationsOperator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="fetch_annotations",
            label="Fetch Annotations",
        )

    def execute(self, ctx: ExecutionContext):

        dataset = ctx.dataset
        view = ctx.view if ctx.view else dataset

        annotations = []

        ctx.log(f"Processing {len(view)} samples for annotations...")

        for sample in view.iter_samples():
            sample_id = str(sample.id)

            # Iterate through all fields in the sample
            for field_name, field_value in sample.iter_fields():

                # Handle Detections
                if isinstance(field_value, fo.Detections):
                    for idx, detection in enumerate(field_value.detections):
                        annotations.append(
                            {
                                "id": f"{sample_id}-{field_name}-{idx}",
                                "sampleId": sample_id,
                                "type": "Detection",
                                "label": detection.label,
                                "confidence": (
                                    round(detection.confidence, 3)
                                    if detection.confidence
                                    else "N/A"
                                ),
                                "field": field_name,
                            }
                        )

        ctx.log(f"Found {len(annotations)} total annotations")

        return {
            "annotations": annotations,
            "total_count": len(annotations),
            "message": f"Found {len(annotations)} annotations",
        }


class GetProgressOperator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_progress",
            label="Get Progress",
        )

    def execute(self, ctx: ExecutionContext):
        operation_name = ctx.params.get("operation_name")
        dataset = ctx.dataset

        if operation_name:
            # Return specific operation progress
            progress_key = f"{operation_name}_progress"
            if progress_key in dataset.info:
                return dataset.info[progress_key]
            else:
                return {"progress": 0, "label": "No operation running"}
        else:
            # Return all active operations
            active_operations = {}
            for key, value in dataset.info.items():
                if key.endswith("_progress"):
                    operation_name = key.replace("_progress", "")
                    active_operations[operation_name] = value

            return {"active_operations": active_operations}


class ComputeObjectEmbeddingsOperator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_embeddings",
            label="Compute Embeddings",
        )

    def execute(self, ctx: foo.ExecutionContext):
        """Compute embeddings on all annotations using FiftyOne patches."""
        dataset = ctx.dataset

        update_progress(ctx, "embedding", 0, "Loading zoo model...")

        try:
            # Use FiftyOne zoo model instead of custom implementation
            import fiftyone.zoo as foz

            # Load a model that supports embeddings
            model = foz.load_zoo_model("clip-vit-base32-torch")

            if not model.has_embeddings:
                return {
                    "num_embeddings": 0,
                    "message": "Selected model does not support embeddings",
                }

            update_progress(ctx, "embedding", 0.1, "Validating dataset...")

            # First check if dataset has detections
            sample_count = len(dataset)
            if sample_count == 0:
                return {
                    "num_embeddings": 0,
                    "message": "Dataset is empty",
                }

            update_progress(ctx, "embedding", 0.2, "Creating patches view...")

            try:
                # Create patches view from detections
                patches_view = dataset.to_patches("detections")
                ctx.log(f"Created patches view with {len(patches_view)} patches")
            except Exception as patches_error:
                ctx.log(f"Failed to create patches view: {str(patches_error)}")
                # Fallback to original custom approach if patches fail
                ctx.log("Falling back to custom embedding computation...")

                update_progress(ctx, "embedding", 0.3, "Using fallback embedding method...")

                # Import the old custom embeddings approach
                try:
                    from .python.embeddings import ModelEmbedder

                    # Use the original working approach
                    embedder = ModelEmbedder.get_embedder("dinov2")

                    # Use the view instead of full dataset if available
                    view = ctx.view if ctx.view else dataset

                    total_embeddings = embedder.embed_detections(
                        samples=view,
                        callback=lambda progress, message: update_progress(
                            ctx, "embedding", 0.3 + (progress * 0.6), message
                        ),
                    )

                    return {
                        "num_embeddings": total_embeddings,
                        "message": f"Computed embeddings for {total_embeddings} objects using fallback method",
                    }

                except ImportError:
                    return {
                        "num_embeddings": 0,
                        "message": f"Patches method failed and fallback not available: {str(patches_error)}",
                    }

            if len(patches_view) == 0:
                return {
                    "num_embeddings": 0,
                    "message": "No detection patches found in dataset",
                }

            update_progress(ctx, "embedding", 0.4, f"Computing embeddings for {len(patches_view)} patches...")

            # Compute embeddings on the patches view
            embeddings = patches_view.compute_embeddings(
                model,
                embeddings_field="embedding"
            )

            total_embeddings = len(patches_view)
            ctx.log(f"Computed embeddings for {total_embeddings} patches")

            update_progress(
                ctx,
                "embedding",
                1,
                f"Completed embedding {total_embeddings} objects in dataset {dataset.name}",
            )

            return {
                "num_embeddings": total_embeddings,
                "message": f"Computed embeddings for {total_embeddings} objects using FiftyOne patches",
            }

        except Exception as e:
            ctx.log(f"Error computing embeddings: {str(e)}")
            return {
                "num_embeddings": 0,
                "message": f"Failed to compute embeddings: {str(e)}",
            }


class ClusterObjectEmbeddingsOperator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="cluster_embeddings",
            label="Cluster Embeddings",
        )

    def execute(self, ctx: foo.ExecutionContext):
        """Cluster embeddings on all annotations using FiftyOne patches."""
        dataset = ctx.dataset

        update_progress(ctx, "clustering", 0, "Initializing clustering...")

        try:
            import hdbscan
            import numpy as np

            min_cluster_size = 5
            epsilon = 0.01

            embeddings = []
            detection_metadata = []

            update_progress(ctx, "clustering", 0.1, "Collecting embeddings from patches...")

            # Collect embeddings from all detections (same as before for compatibility)
            for sample in dataset:
                if not hasattr(sample, "detections") or sample.detections is None:
                    continue
                for idx, detection in enumerate(sample.detections.detections):
                    if hasattr(detection, "embedding") and detection.embedding is not None:
                        embeddings.append(np.array(detection.embedding))
                        detection_metadata.append({
                            "sample_id": str(sample.id),
                            "detection_idx": idx,
                            "label": detection.label,
                        })

            if len(embeddings) == 0:
                return {
                    "num_clusters": 0,
                    "labels": [],
                    "message": "No embeddings found. Please compute embeddings first.",
                }

            update_progress(ctx, "clustering", 0.3, f"Clustering {len(embeddings)} embeddings...")

            # Convert to numpy array for clustering
            embeddings_array = np.array(embeddings)

            clusterer = hdbscan.HDBSCAN(
                min_samples=min_cluster_size // 2,
                min_cluster_size=min_cluster_size,
                cluster_selection_epsilon=epsilon,
                cluster_selection_method="leaf",
            )

            update_progress(ctx, "clustering", 0.8, "Fitting clustering model...")
            labels = clusterer.fit_predict(embeddings_array)

            update_progress(
                ctx,
                "clustering",
                1,
                f"Completed clustering objects in dataset {dataset.name}",
            )

            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            # Optional: Store cluster labels back to detections
            for i, detection_meta in enumerate(detection_metadata):
                sample = dataset[detection_meta["sample_id"]]
                detection = sample.detections.detections[detection_meta["detection_idx"]]
                detection.cluster_id = int(labels[i])
                sample.save()

            return {
                "num_clusters": num_clusters,
                "labels": labels.tolist(),
                "message": f"Clustered objects into {num_clusters} clusters using FiftyOne patches embeddings",
            }

        except Exception as e:
            ctx.log(f"Error clustering embeddings: {str(e)}")
            return {
                "num_clusters": 0,
                "labels": [],
                "message": f"Failed to cluster embeddings: {str(e)}",
            }


def register(plugin):
    plugin.register(FetchAnnotationsOperator)
    plugin.register(GetProgressOperator)
    plugin.register(ComputeObjectEmbeddingsOperator)
    plugin.register(ClusterObjectEmbeddingsOperator)
