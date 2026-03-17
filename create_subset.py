import os
import shutil
from pathlib import Path

def create_subset():
    data_dir = Path("/home/amma/Documents/data_netravaad/dataset")
    img_dir = data_dir / "images"
    label_dir = data_dir / "labels"

    # Match the exact sorting and filtering from dataset.py
    all_images = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])

    total = len(all_images)
    keep = max(1, int(total * 0.03))
    
    subset_images = all_images[:keep]
    print(f"Selected {len(subset_images)} images out of {total}.")

    subset_dir = Path("dataset_subset")
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    
    subset_dir.mkdir(exist_ok=True)
    (subset_dir / "images").mkdir(exist_ok=True)
    (subset_dir / "labels").mkdir(exist_ok=True)

    print("Copying files...")
    for img_path in subset_images:
        shutil.copy(img_path, subset_dir / "images" / img_path.name)
        
        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(label_path, subset_dir / "labels" / label_path.name)

    print("Creating zip archive...")
    shutil.make_archive("dataset_subset_0.06", "zip", subset_dir)
    print("Done! Created dataset_subset_0.06.zip in the current directory.")

if __name__ == "__main__":
    create_subset()
