from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import torch
import time
from tqdm import tqdm


CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute.")
        return result

    return wrapper


def get_padded_bbox_crop(
    image: Image.Image,
    bbox: torch.Tensor,
    padding: int = 48,
    output_size: int = 128,
) -> Image.Image:
    x1, y1, x2, y2 = bbox.tolist()
    long_edge = torch.argmax(torch.tensor([x2 - x1, y2 - y1]))  # 0 is w, 1 is h
    if long_edge == 0:
        x1 -= padding
        x2 += padding
        short_edge_padding = torch.div(
            (x2 - x1) - (y2 - y1), 2, rounding_mode="floor"
        ).item()
        y1 -= short_edge_padding
        y2 += short_edge_padding
    else:
        y1 -= padding
        y2 += padding
        short_edge_padding = torch.div(
            ((y2 - y1) - (x2 - x1)), 2, rounding_mode="floor"
        ).item()
        x1 -= short_edge_padding
        x2 += short_edge_padding
    padded_bbox = torch.tensor([x1, y1, x2, y2])
    return image.crop(padded_bbox.tolist()).resize((output_size, output_size))


def get_padded_bbox_crop_and_rect(
    image: Image.Image,
    bbox: torch.Tensor,
    padding: int = 48,
    output_size: int = 128,
):
    """Return the padded, square crop resized to output_size and the rectangle
    coordinates of the original bbox within that resized crop.

    The rectangle coordinates are a tuple (x1, y1, x2, y2) in the crop's
    coordinate system [0, output_size].
    """
    x1, y1, x2, y2 = bbox.tolist()
    # Determine which edge is longer to apply fixed padding and square padding
    long_edge = torch.argmax(
        torch.tensor([x2 - x1, y2 - y1])
    )  # 0 is width, 1 is height
    px1, py1, px2, py2 = x1, y1, x2, y2
    if long_edge == 0:
        px1 -= padding
        px2 += padding
        short_edge_padding = torch.div(
            (px2 - px1) - (py2 - py1), 2, rounding_mode="floor"
        ).item()
        py1 -= short_edge_padding
        py2 += short_edge_padding
    else:
        py1 -= padding
        py2 += padding
        short_edge_padding = torch.div(
            (py2 - py1) - (px2 - px1), 2, rounding_mode="floor"
        ).item()
        px1 -= short_edge_padding
        px2 += short_edge_padding

    padded_w = px2 - px1
    padded_h = py2 - py1
    # Create the crop from the padded square region and resize
    crop = image.crop([px1, py1, px2, py2]).resize((output_size, output_size))

    # Map original bbox into the padded crop coordinate frame, then scale
    rel_x1 = x1 - px1
    rel_y1 = y1 - py1
    rel_x2 = x2 - px1
    rel_y2 = y2 - py1

    # Use per-axis scale in case rounding makes padded_w != padded_h by 1px
    scale_x = output_size / padded_w if padded_w != 0 else 1.0
    scale_y = output_size / padded_h if padded_h != 0 else 1.0

    rx1 = int(round(rel_x1 * scale_x))
    ry1 = int(round(rel_y1 * scale_y))
    rx2 = int(round(rel_x2 * scale_x))
    ry2 = int(round(rel_y2 * scale_y))

    # Clamp to valid drawable area [0, output_size-1]
    rx1 = max(0, min(output_size - 1, rx1))
    ry1 = max(0, min(output_size - 1, ry1))
    rx2 = max(0, min(output_size - 1, rx2))
    ry2 = max(0, min(output_size - 1, ry2))

    # Ensure proper ordering
    if rx1 > rx2:
        rx1, rx2 = rx2, rx1
    if ry1 > ry2:
        ry1, ry2 = ry2, ry1

    rect = (rx1, ry1, rx2, ry2)

    return crop, rect


import numpy as np
from pathlib import Path
from collections import defaultdict


def create_spritesheets(
    annotations: list[dict],
    images: list[dict],
    mods_dir: Path,
    n_rows: int = 16,
    n_cols: int = 16,
    thumbnail_size: int = 64,
):
    SPRITESHEET_CACHE_DIR = CACHE_DIR / "spritesheets"
    SPRITESHEET_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if (SPRITESHEET_CACHE_DIR / "spritesheet_0.png").exists():
        return

    # Build lookup: image_id -> list of its annotations
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    # Build lookup: image_id -> file path
    image_paths = {img["id"]: str(mods_dir / img["file_name"]) for img in images}

    crops = []

    for image_id, anns in tqdm(anns_by_image.items()):
        image_path = mods_dir / image_paths[image_id]
        try:
            with Image.open(image_path) as img:
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    bbox = np.array([x, y, x + w, y + h])

                    try:
                        crop = img.crop((x, y, x + w, y + h))
                        crop = crop.resize((thumbnail_size, thumbnail_size))
                        # crop, rect = get_padded_bbox_crop_and_rect(
                        #     img,
                        #     bbox=bbox,
                        #     padding=48,
                        #     output_size=thumbnail_size,
                        # )

                        # # Draw rectangle around the original bbox
                        # draw = ImageDraw.Draw(crop)
                        # draw.rectangle(rect, outline=(255, 255, 255), width=1)

                        # # Add text showing bbox dimensions
                        # try:
                        #     from PIL import ImageFont

                        #     font = ImageFont.truetype("arial.ttf", size=12)
                        # except:
                        #     # Fallback to default font if arial.ttf not available
                        #     font = ImageFont.load_default()

                        # bbox_text = f"{int(w)},{int(h)}"
                        # draw.text(
                        #     (thumbnail_size - 2, 2),  # near the top right
                        #     bbox_text,
                        #     fill=(255, 0, 0),
                        #     font=font,
                        #     anchor="rt",  # right-top anchoring
                        # )

                        # Convert PIL image to numpy array for consistency
                        crop_array = np.array(crop)
                        crops.append(crop_array)

                    except Exception as e:
                        # Fallback to black tile if any error occurs
                        print(f"Error processing annotation {ann['id']}: {e}")
                        crop_array = np.zeros(
                            (thumbnail_size, thumbnail_size, 3), dtype=np.uint8
                        )
                        crops.append(crop_array)

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create black tiles for all annotations in this image
            for _ in anns:
                crop_array = np.zeros(
                    (thumbnail_size, thumbnail_size, 3), dtype=np.uint8
                )
                crops.append(crop_array)

    # Assemble crops into spritesheets
    max_per_sheet = n_rows * n_cols
    current_index = 0
    sheet_index = 0

    while current_index < len(crops):
        spritesheet = np.zeros(
            (n_rows * thumbnail_size, n_cols * thumbnail_size, 3), dtype=np.uint8
        )

        for i in range(max_per_sheet):
            if current_index >= len(crops):
                break
            crop = crops[current_index]
            row = i // n_cols
            col = i % n_cols
            y0, y1 = row * thumbnail_size, (row + 1) * thumbnail_size
            x0, x1 = col * thumbnail_size, (col + 1) * thumbnail_size
            spritesheet[y0:y1, x0:x1] = crop
            current_index += 1

        out_path = str(SPRITESHEET_CACHE_DIR / f"spritesheet_{sheet_index}.png")
        # Convert numpy array back to PIL Image for saving
        spritesheet_img = Image.fromarray(spritesheet)
        spritesheet_img.save(out_path)
        sheet_index += 1
