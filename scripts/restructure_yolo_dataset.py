import os
import shutil
import random

# ====== CONFIG ======
BASE_DIR = (
    r"D:\Desktop\AlinasPrograms\Projects\Foodalyze\data\IndianFoodDatasetFinalFiltered"
)
TRAIN_SPLIT = 0.8  # 80% train
VAL_SPLIT = 0.1  # 10% val
TEST_SPLIT = 0.1  # 10% test
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")  # accepted image types
# =====================

# create output folders
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(BASE_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, split, "labels"), exist_ok=True)

# get all image paths
all_images = []
for food_class in os.listdir(BASE_DIR):
    class_path = os.path.join(BASE_DIR, food_class)
    if not os.path.isdir(class_path) or food_class in ["train", "valid", "test"]:
        continue
    for file in os.listdir(class_path):
        if file.lower().endswith(IMAGE_EXTENSIONS):
            all_images.append(os.path.join(class_path, file))

# shuffle and split
random.shuffle(all_images)
n_total = len(all_images)
n_train = int(TRAIN_SPLIT * n_total)
n_val = int(VAL_SPLIT * n_total)
train_imgs = all_images[:n_train]
val_imgs = all_images[n_train : n_train + n_val]
test_imgs = all_images[n_train + n_val :]


def copy_files(img_list, split):
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        label_path = os.path.splitext(img_path)[0] + ".txt"  # expected YOLO label
        dst_img = os.path.join(BASE_DIR, split, "images", img_name)
        dst_lbl = os.path.join(BASE_DIR, split, "labels", os.path.basename(label_path))

        # copy image
        shutil.copy(img_path, dst_img)

        # copy label if it exists
        if os.path.exists(label_path):
            shutil.copy(label_path, dst_lbl)

        # optional: print progress
        print(f"Copied: {img_name} → {split}")


# copy images and labels
copy_files(train_imgs, "train")
copy_files(val_imgs, "valid")
copy_files(test_imgs, "test")

print("\nDataset restructured successfully!")
print(
    f"Total images: {n_total} | train: {len(train_imgs)}, val: {len(val_imgs)}, test: {len(test_imgs)}"
)
print(f"Check folders: {BASE_DIR}\\train, {BASE_DIR}\\valid, {BASE_DIR}\\test")
