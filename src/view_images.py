import os
import cv2
import matplotlib.pyplot as plt

# Change this path if you want to see another category
image_folder = "data/real_dataset/animals"

# Get first image
image_name = os.listdir(image_folder)[0]

image_path = os.path.join(image_folder, image_name)

# Read image
image = cv2.imread(image_path)

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show image
plt.imshow(image)
plt.title(image_name)
plt.axis("off")

plt.show()