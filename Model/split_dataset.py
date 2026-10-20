import os
import shutil
import random

# ------------------------------------------
# PATHS - UPDATE THESE IF NEEDED
# ------------------------------------------

source_path = r"D:\Documents\Datasets\Animal_Classi\Animal_20"
dest_path = r"D:\AI PROJECTS\Animal_Classification\Data"

# ------------------------------------------

# Make sure main folders exist
train_path = os.path.join(dest_path, "Train")
val_path = os.path.join(dest_path, "Val")
test_path = os.path.join(dest_path, "Test")

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Loop through all 20 classes
classes = os.listdir(source_path)

for cls in classes:
    cls_folder = os.path.join(source_path, cls)
    images = os.listdir(cls_folder)

    # filter only images
    images = [img for img in images if img.lower().endswith((".jpg", ".png", ".jpeg"))]

    random.shuffle(images)

    total = len(images)
    train_split = int(0.7 * total)
    val_split = int(0.15 * total)

    train_imgs = images[:train_split]
    val_imgs = images[train_split:train_split + val_split]
    test_imgs = images[train_split + val_split:]

    # Create class folders
    os.makedirs(os.path.join(train_path, cls), exist_ok=True)
    os.makedirs(os.path.join(val_path, cls), exist_ok=True)
    os.makedirs(os.path.join(test_path, cls), exist_ok=True)

    # Move images
    for img in train_imgs:
        shutil.copy(os.path.join(cls_folder, img), os.path.join(train_path, cls, img))

    for img in val_imgs:
        shutil.copy(os.path.join(cls_folder, img), os.path.join(val_path, cls, img))

    for img in test_imgs:
        shutil.copy(os.path.join(cls_folder, img), os.path.join(test_path, cls, img))

    print(f"{cls}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

print("\nDataset split completed successfully!")
