import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os


def plot_plankton_by_id(data_dir, id_img=0, annotations_file='_annotations.coco.json', LABEL=True):
    with open(os.path.join(data_dir, annotations_file), 'r') as f:
        coco_annotations = json.load(f)

    # Extract image file paths and mask annotations from the COCO annotations
    images = coco_annotations['images']
    annotations = coco_annotations['annotations']
    categories = coco_annotations['categories']

    # Specify the index of the image you want to display
    image_index = id_img  # Replace with the desired index

    if 0 <= image_index < len(images):
        # Get the image information at the specified index
        image_info = images[image_index]

        # Read the image using cv2
        image_file = image_info['file_name']
        image = cv2.imread(os.path.join(data_dir, image_file))

        # Create a mask with the same size as the image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Find the corresponding mask annotations
        image_id = image_info['id']
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
        for annotation in image_annotations:
            mask_annotations = annotation['segmentation']
            mask_annotations = np.array(mask_annotations, dtype=np.int32)
            mask_annotations = np.reshape(mask_annotations, (-1, 2))
            cv2.fillPoly(mask, [mask_annotations], 255)

        # Create a copy of the image for overlay
        image_with_mask = image.copy()

        # Draw blue contours on the image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_with_mask, contours, -1, (255, 0, 0), 2)

        # Convert the images to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        image_with_mask_rgb = cv2.cvtColor(image_with_mask, cv2.COLOR_BGR2RGB)

        # Get the class names associated with the masks
        class_names = [categories[annotation['category_id'] - 1]['name'] for annotation in image_annotations]

        # Display the images with class names
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask_rgb)
        axes[1].set_title('Mask')
        axes[1].axis('off')

        axes[2].imshow(image_with_mask_rgb)
        axes[2].set_title('Original Image + Mask')
        axes[2].axis('off')

        if LABEL:
            # Add the class names as text on the last image
            for class_name, contour in zip(class_names, contours):
                x, y, _, _ = cv2.boundingRect(contour)
                axes[2].text(x, y, class_name, color='white', backgroundcolor='black', fontsize=8)

        plt.tight_layout()
        plt.show()
    else:
        print(f"Invalid image index. There are {len(images)} images in the dataset.")


def plot_3_examples(data_dir, annotations_file='_annotations.coco.json', LABEL=True):
    with open(os.path.join(data_dir, annotations_file), 'r') as f:
        coco_annotations = json.load(f)

    # Extract image file paths and mask annotations from the COCO annotations
    images = coco_annotations['images']
    annotations = coco_annotations['annotations']
    categories = coco_annotations['categories']

    # Randomly select 3 images
    random_images = random.sample(images, 3)

    # Display images and masks
    for image_info in random_images:
        # Read the image using cv2
        image_file = image_info['file_name']
        image = cv2.imread(os.path.join(data_dir, image_file))

        # Create a mask with the same size as the image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Find the corresponding mask annotations
        image_id = image_info['id']
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
        for annotation in image_annotations:
            mask_annotations = annotation['segmentation']
            mask_annotations = np.array(mask_annotations, dtype=np.int32)
            mask_annotations = np.reshape(mask_annotations, (-1, 2))
            cv2.fillPoly(mask, [mask_annotations], 255)

        # Create a copy of the image for overlay
        image_with_mask = image.copy()

        # Draw blue contours on the image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_with_mask, contours, -1, (255, 0, 0), 2)

        # Convert the images to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        image_with_mask_rgb = cv2.cvtColor(image_with_mask, cv2.COLOR_BGR2RGB)

        # Get the class names associated with the masks
        class_names = [categories[annotation['category_id'] - 1]['name'] for annotation in image_annotations]

        # Display the images with class names
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask_rgb)
        axes[1].set_title('Mask')
        axes[1].axis('off')

        axes[2].imshow(image_with_mask_rgb)
        axes[2].set_title('Original Image + Mask')
        axes[2].axis('off')

        # Add the class names as text on the last image
        for class_name, contour in zip(class_names, contours):
            x, y, _, _ = cv2.boundingRect(contour)
            axes[2].text(x, y, class_name, color='white', backgroundcolor='black', fontsize=8)

        plt.tight_layout()
        plt.show()
