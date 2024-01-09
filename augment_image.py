# augment_image
import os
import pandas as pd
from PIL import Image, ImageOps
import random

# Assuming image_df_full is already loaded with 'Filepath' and 'label' columns

# Analyze label distribution
label_counts = image_df_full['label'].value_counts()
max_count = label_counts.max()

# Root directory for augmented images
root_save_dir = 'data/augmented_images'

# Function to augment an image
def augment_image(image_path, save_path):
    # Load the original image
    image = Image.open(image_path)

    # Apply some transformations (you can customize these)
    if random.choice([True, False]):
        image = ImageOps.mirror(image)  # Horizontal flip
    if random.choice([True, False]):
        image = ImageOps.flip(image)  # Vertical flip
    if random.choice([True, False]):
        image = image.rotate(random.randint(-30, 30))  # Rotation

    # Save the augmented image
    image.save(save_path)

# Augment images for each label
for label, count in label_counts.iteritems():
    # Calculate how many images to augment
    num_to_augment = max_count - count

    # Create a sub-folder for each label
    label_save_dir = os.path.join(root_save_dir, label)
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    # Get image paths for the label
    image_paths = image_df_full[image_df_full['label'] == label]['Filepath']

    # Augment images
    for i in range(num_to_augment):
        # Select a random image to augment
        image_path = random.choice(image_paths.tolist())
        save_path = os.path.join(label_save_dir, f'{i}.jpg')
        augment_image(image_path, save_path)
