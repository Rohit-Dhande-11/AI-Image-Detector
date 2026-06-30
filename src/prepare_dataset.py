import os
import random
import shutil

random.seed(42)

REAL_PATH = "data/real_dataset"
FAKE_PATH = "data/Ai_generated_dataset"
OUTPUT_PATH = "processed_data"

# Remove old processed dataset
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)

# Create folders
for folder in [
    "train/real",
    "train/fake",
    "validation/real",
    "validation/fake",
    "test/real",
    "test/fake",
]:
    os.makedirs(os.path.join(OUTPUT_PATH, folder), exist_ok=True)


def collect_images(folder):
    images = []

    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)

        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(os.path.join(category_path, file))

    return images


real_images = collect_images(REAL_PATH)
fake_images = collect_images(FAKE_PATH)


def split_and_copy(images, label):

    random.shuffle(images)

    total = len(images)

    train = int(total * 0.8)
    validation = int(total * 0.1)

    train_images = images[:train]
    validation_images = images[train:train + validation]
    test_images = images[train + validation:]

    # Copy training images
    for i, img in enumerate(train_images):
        extension = os.path.splitext(img)[1]
        new_name = f"{label}_train_{i}{extension}"
        shutil.copy(img, os.path.join(OUTPUT_PATH, "train", label, new_name))

    # Copy validation images
    for i, img in enumerate(validation_images):
        extension = os.path.splitext(img)[1]
        new_name = f"{label}_validation_{i}{extension}"
        shutil.copy(img, os.path.join(OUTPUT_PATH, "validation", label, new_name))

    # Copy test images
    for i, img in enumerate(test_images):
        extension = os.path.splitext(img)[1]
        new_name = f"{label}_test_{i}{extension}"
        shutil.copy(img, os.path.join(OUTPUT_PATH, "test", label, new_name))

    print(f"\n{label}:")
    print(f" Train      : {len(train_images)}")
    print(f" Validation : {len(validation_images)}")
    print(f" Test       : {len(test_images)}")


split_and_copy(real_images, "real")
split_and_copy(fake_images, "fake")

print("\n✅ Dataset prepared successfully!")