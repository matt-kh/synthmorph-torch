import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as scind
from PIL import Image
from prettytable import PrettyTable


def torch_model_parameters(model):
    table = PrettyTable(['Module','Parameter'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f'Total Trainable Params: {total_params}')
    return total_params


def image_to_numpy(path):
    image = Image.open(path)
    numpy_image = np.array(image)
    return numpy_image


def invert_grayscale(gray_image):
    max_intensity = np.max(gray_image)
    inverted_image = max_intensity - gray_image
    
    return inverted_image


def binarize(image, threshold):
    binary_image = np.where(image > threshold, 1, 0)
    return binary_image


def overlay_images(image1, image2, alpha=0.5, pixel_range=(0, 1)):
    blended_image = alpha * image1 + (1 - alpha) * image2
    blended_image = np.clip(blended_image, pixel_range[0], pixel_range[1]).astype(np.float32)
    return blended_image


def plot_array_row(array_row, headers, cmap='gray'):
    num_images = len(array_row)

    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))

    for i, (image, header) in enumerate(zip(array_row, headers)):
        axes[i].imshow(image, cmap=cmap)
        axes[i].axis('off')
        axes[i].set_title(header)  # Set the title/header for each subplot

    plt.tight_layout()
    plt.show()
    
    
def convert_to_single_rgb(grayscale_image, channel='red'):
    x, y = grayscale_image.shape
    
    # Create an empty RGB image
    rgb_image = np.zeros((x, y, 3), dtype=np.uint8)
    
    if channel == 'red':
        rgb_image[:, :, 0] = grayscale_image  # Assign grayscale values to the red channel
    elif channel == 'green':
        rgb_image[:, :, 1] = grayscale_image  # Assign grayscale values to the green channel
    elif channel == 'blue':
        rgb_image[:, :, 2] = grayscale_image  # Assign grayscale values to the blue channel
    
    return rgb_image


def rotate(image, angle):
    return scind.rotate(image, angle, reshape=False, mode='nearest')


def superimpose_circles(
    image, 
    pixel_value=255, 
    size_range=(2, 4), 
    dist_range=(50, 81), 
    rotate=0,
    x_shift = 0,
    y_shift=0,
    circle_thickness=-1,
):
    height, width = image.shape
    center_x = width // 2
    center_y = height // 2
    pixel_value = pixel_value / 255
    # shift as percentage
    x_shift = int(x_shift * width / 100)
    y_shift = int(y_shift * height / 100)
    
    # Generate random percentages for height and width
    percent_height = np.random.randint(*dist_range)
    percent_width = np.random.randint(*dist_range)

    # Calculate the radius based on the random percentages and minimum dimension
    min_dimension = min(height, width)
    min_radius = min_dimension * np.random.uniform(*size_range)

    distance_x = width * (percent_width / 100)
    distance_y = height * (percent_height / 100)
    radius = int(min_radius)
    angle = abs(1 - rotate/90)   # rotation maximum of 180 degrees
    # Create a copy of the input image
    result = np.copy(image)
    
    # Draw the four circles  such that they are the corners of a rectangle
    cv2.circle(
        result, 
        (center_x - int(distance_x*angle/2) + x_shift,\
            center_y - int(distance_y/2) + y_shift), \
        radius, pixel_value, circle_thickness, \
    )
    
    cv2.circle(
        result, 
        (center_x + int(distance_x/2) + x_shift, \
            center_y - int(distance_y*angle/2) + y_shift), \
        radius, pixel_value, circle_thickness, \
    )
    
    cv2.circle(
        result, 
        (center_x - int(distance_x/2) + x_shift, \
            center_y + int(distance_y*angle/2) + y_shift), \
        radius, pixel_value, circle_thickness, \
    )
    
    cv2.circle(
        result, 
        (center_x + int(distance_x*angle/2) + x_shift, \
            center_y + int(distance_y/2) + y_shift), \
        radius, pixel_value, circle_thickness, \
    )
        
    return result