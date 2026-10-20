import os
import shutil

# Source folder that contains 90 animal folders
source_path = r"D:\Documents\Datasets\Animal_Classi\archive\animals\animals"

# Destination folder where 20 selected classes will be copied
destination_path = r"D:\Documents\Datasets\Animal_Classi\Animal_20"

# ------------------------------------

# The 20 selected classes
selected_classes = [
    "lion", "tiger", "leopard", "deer", "wolf",
    "dog", "cat", "horse", "cow", "goat",
    "sheep", "pig", "owl", "eagle", "parrot",
    "crow", "dolphin", "shark", "whale", "chimpanzee"
]


# Create destination folder if not exists
os.makedirs(destination_path, exist_ok=True)

for cls in selected_classes:
    src = os.path.join(source_path, cls)
    dst = os.path.join(destination_path, cls)

    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"[OK] Copied: {cls}")
    else:
        print(f"[MISSING] Class not found in dataset: {cls}")

print("\nExtraction Completed Successfully!")
