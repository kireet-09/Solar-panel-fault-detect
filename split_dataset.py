import os
import shutil
import random

SOURCE_DIR = "Faulty_solar_panel"
DEST_DIR = "dataset"

SPLIT_RATIO = 0.8

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        random.shuffle(images)
        split_index = int(len(images) * SPLIT_RATIO)

        train_images = images[:split_index]
        test_images = images[split_index:]

        for subset, subset_images in zip(['train', 'test'], [train_images, test_images]):
            subset_dir = os.path.join(DEST_DIR, subset, class_name)
            os.makedirs(subset_dir, exist_ok=True)
            for img in subset_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(subset_dir, img)
                if os.path.isfile(src):  # ✅ Skip folders or non-files
                    shutil.copy2(src, dst)

print("Dataset split complete.")
