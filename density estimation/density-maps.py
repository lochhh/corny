import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

def read_yolo_annotations(annotation_file, image_width, image_height):
    """
    Read YOLO format annotations and convert to pixel coordinates.
    
    :param annotation_file: Path to YOLO annotation file
    :param image_width: Width of the image
    :param image_height: Height of the image
    :return: List of (x, y, class) tuples
    """
    points = []
    with open(annotation_file, 'r') as f:
        for line in f:
            class_id, x_center, y_center, _, _ = map(float, line.strip().split())
            x = int(x_center * image_width)
            y = int(y_center * image_height)
            points.append((x, y, int(class_id)))
    return points

def create_density_map(image_shape, points, sigma=10):
    """
    Create a density map from point annotations.
    
    :param image_shape: Tuple of (height, width) of the image
    :param points: List of (x, y, class) tuples
    :param sigma: Standard deviation for Gaussian kernel
    :return: Density map as a 2D numpy array
    """
    density_map = np.zeros(image_shape, dtype=np.float32)
    
    for x, y, _ in points:
        density_map[y, x] = 1
    
    density_map = gaussian_filter(density_map, sigma=sigma, mode='constant')
    
    # Normalize the map
    density_map = density_map / density_map.sum() * len(points)
    
    return density_map

def create_class_specific_density_maps(image_shape, points, class_labels, sigma=10):
    """
    Create separate density maps for each class.
    
    :param image_shape: Tuple of (height, width) of the image
    :param points: List of (x, y, class) tuples
    :param num_classes: Number of classes in the dataset
    :param sigma: Standard deviation for Gaussian kernel
    :return: List of density maps, one for each class
    """
    class_density_maps = []
    
    for class_id in class_labels:
        class_points = [(x, y, c) for x, y, c in points if c == class_id ]
        print(class_points)
        class_map = create_density_map(image_shape, class_points, sigma)
        class_density_maps.append(class_map)

    return class_density_maps

def resize_image(image, target_size):
    """
    Resize image while maintaining aspect ratio.
    
    :param image: PIL Image object
    :param target_size: Tuple of (width, height) for the target size
    :return: Resized PIL Image object
    """
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    target_width, target_height = target_size

    if aspect_ratio > target_width / target_height:
        # Image is wider than target aspect ratio
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        # Image is taller than target aspect ratio
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a new image with the target size and paste the resized image
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

# def process_images(image_folder, annotation_folder, output_folder, class_labels, target_size=(256, 256), sigma=10):
#     """
#     Process all images in the given folder and create density maps.
    
#     :param image_folder: Path to folder containing images
#     :param annotation_folder: Path to folder containing YOLO annotations
#     :param output_folder: Path to save output density maps
#     :param num_classes: Number of classes in the dataset
#     :param sigma: Standard deviation for Gaussian kernel
#     """
#     os.makedirs(output_folder, exist_ok=True)
    
#     for filename in os.listdir(image_folder):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image_path = os.path.join(image_folder, filename)
#             annotation_path = os.path.join(annotation_folder, os.path.splitext(filename)[0] + '.txt')
            
#             if not os.path.exists(annotation_path):
#                 print(f"Annotation file not found for {filename}, skipping.")
#                 continue
            
#             print(f"Processing {filename}")

#             # Load image and get dimensions
#             image = Image.open(image_path)
#             image_width, image_height = image.size
            
#             # Read YOLO annotations
#             points = read_yolo_annotations(annotation_path, image_width, image_height)
            
#             # Create overall density map
#             # overall_density_map = create_density_map((image_height, image_width), points, sigma)
            
#             # Create class-specific density maps
#             class_density_maps = create_class_specific_density_maps((image_height, image_width), points, class_labels, sigma)
            
#             # Save density maps
#             # np.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_overall_density.npy"), overall_density_map)
#             for i, class_map in enumerate(class_density_maps):
#                 np.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_class_{i}_density.npy"), class_map)
            
#             print(f"Processed {filename}")

def process_images(image_folder, annotation_folder, output_folder, class_labels, target_size=(256, 256), sigma=10):
    """
    Process all images in the given folder, resize them, and create density maps.
    
    :param image_folder: Path to folder containing images
    :param annotation_folder: Path to folder containing YOLO annotations
    :param output_folder: Path to save output density maps
    :param class_labels: List of class labels
    :param target_size: Tuple of (width, height) for the target size
    :param sigma: Standard deviation for Gaussian kernel
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a 'resized' subfolder in the input image folder
    resized_folder = os.path.join(image_folder, 'resized')
    os.makedirs(resized_folder, exist_ok=True)
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            annotation_path = os.path.join(annotation_folder, os.path.splitext(filename)[0] + '.txt')
            
            if not os.path.exists(annotation_path):
                print(f"Annotation file not found for {filename}, skipping.")
                continue
            
            print(f"Processing {filename}")

            # Load and resize image
            original_image = Image.open(image_path)
            resized_image = resize_image(original_image, target_size)
            image_width, image_height = resized_image.size
            
            # Read YOLO annotations and adjust for resized image
            original_width, original_height = original_image.size
            points = read_yolo_annotations(annotation_path, original_width, original_height)
            
            # Adjust point coordinates for resized image
            scale_x = image_width / original_width
            scale_y = image_height / original_height
            adjusted_points = [(int(x * scale_x), int(y * scale_y), c) for x, y, c in points]
            
            # Create class-specific density maps
            class_density_maps = create_class_specific_density_maps((image_height, image_width), adjusted_points, class_labels, sigma)
            
            # Save resized image in the 'resized' subfolder
            resized_image_path = os.path.join(resized_folder, filename)
            resized_image.save(resized_image_path)
            
            # Save density maps
            for i, class_map in enumerate(class_density_maps):
                np.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_class_{i}_density.npy"), class_map)
            
            print(f"Processed {filename}")

def visualize_density_map(image_path, density_map_path, output_path=None):
    """
    Visualize the original image and its corresponding density map side by side.
    
    :param image_path: path to the original image
    :param density_map_path: path to the density map
    :param output_path: path to save the visualization
    """
    
    image = Image.open(image_path)
    # print(image)
    # Load the density map
    density_map = np.load(density_map_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Display original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Display density map
    im = ax2.imshow(density_map, cmap='jet')
    ax2.set_title("Density Map")
    ax2.axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    # plt.savefig(output_path)
    # plt.close()

if __name__ == "__main__":
    image_folder = '../datasets/corn_yolo_no_segment/images/corn_kernel_train/'
    annotation_folder = '../datasets/corn_yolo_no_segment/labels/corn_kernel_train/'
    output_folder = './maps/kernel-train/'
    class_labels = [ 0 ] # 0 for kernel
    sigma = 6
    target_size = (512, 512)
    
    # process_images(image_folder, annotation_folder, output_folder, class_labels, target_size, sigma)
    
    img_name = 'corn_003'
    image_path = image_folder + 'resized/' + img_name + '.jpg'
    kernel_density_map_path = output_folder + img_name +  '_class_0_density.npy'
    # allclass_density_map_path = './maps/all-classes/' + img_name + '_overall_density.npy'
    # visualize_density_map(image_path, allclass_density_map_path, output_path=None)
    visualize_density_map(image_path, kernel_density_map_path, output_path=None)
    