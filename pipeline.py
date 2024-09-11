import cv2
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb
from skimage.segmentation import flood_fill
import skimage.morphology as morpho
from matplotlib import pyplot as plt
from numpy.lib.function_base import median
import numpy as np


def remove_skin_hair(image, kernel_size):
    hair_kernel_size = 47
    median_blur_image = cv2.medianBlur(image, hair_kernel_size)
    only_hair = cv2.subtract(median_blur_image, image)

    _, mask = cv2.threshold(only_hair, int(0.02*255), 255, cv2.THRESH_BINARY)
    kernel = morpho.disk(2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    # Create a copy of the input image to store the inpainted result
    inpainted_image = image.copy()

    # Get the coordinates of the noisy regions from the binary mask
    noisy_coords = np.where(mask > 0)

    for y, x in zip(*noisy_coords):
        y_start, y_end = max(
            0, y - kernel_size), min(image.shape[0], y + kernel_size + 1)
        x_start, x_end = max(
            0, x - kernel_size), min(image.shape[1], x + kernel_size + 1)

        neighborhood = image[y_start:y_end, x_start:x_end]

        if neighborhood.size > 0:
            inpainted_image[y, x] = np.median(neighborhood)

    return inpainted_image


def find_good_region(image):
    _, good_region = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = morpho.disk(5)

    good_region = cv2.morphologyEx(good_region, cv2.MORPH_OPEN, kernel)
    good_region = cv2.morphologyEx(good_region, cv2.MORPH_CLOSE, kernel)

    flag_painted = 0

    if (good_region[0, 0] == 255):
        regions = flood_fill(good_region, seed_point=(0, 0), new_value=128)
        flag_painted = 1
    if (good_region[image.shape[0]-1, image.shape[1]-1] == 255):
        regions = flood_fill(regions, seed_point=(
            image.shape[0]-1, image.shape[1]-1), new_value=128)
        flag_painted = 1
    if (good_region[0, image.shape[1]-1] == 255):
        regions = flood_fill(regions, seed_point=(
            0, image.shape[1]-1), new_value=128)
        flag_painted = 1
    if (good_region[image.shape[0]-1, 0] == 255):
        regions = flood_fill(regions, seed_point=(
            image.shape[0]-1, 0), new_value=128)
        flag_painted = 1

    if (flag_painted == 0):
        return np.full_like(image, 255, dtype='uint8')

    good_region = (255*(regions == 128)).astype('uint8')

    kernel = morpho.disk(30)
    good_region = cv2.morphologyEx(good_region, cv2.MORPH_DILATE, kernel)
    good_region = 255 - good_region
    return good_region


def segmentation(inpainted, good_region):
    inner_circle_values = inpainted[(good_region == 255)]

    otsu_threshold, _ = cv2.threshold(
        inner_circle_values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print("Obtained threshold: ", otsu_threshold)

    segmentation_mask = np.zeros_like(inpainted)
    segmentation_mask = np.where(good_region == 0, 0, segmentation_mask)
    segmentation_mask = np.where((good_region != 0) & (
        inpainted < otsu_threshold), 255, segmentation_mask)

    # selecting only the region with the biggest area
    kernel = morpho.disk(20)
    segmentation_mask = cv2.morphologyEx(
        segmentation_mask, cv2.MORPH_OPEN, kernel)
    kernel = morpho.disk(30)
    segmentation_mask = cv2.morphologyEx(
        segmentation_mask, cv2.MORPH_DILATE, kernel)
    kernel = morpho.disk(85)
    segmentation_mask = cv2.morphologyEx(
        segmentation_mask, cv2.MORPH_CLOSE, kernel)

    if (np.all(segmentation_mask == 0)):  # check if the mask is null everywhere
        return segmentation_mask

    numlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        segmentation_mask)
    matrix = np.column_stack((range(numlabels), stats[:, 4]))
    sorted_matrix = sorted(matrix, key=lambda row: row[1], reverse=True)

    # draw the biggest contour
    final_mask = (255*(labels == sorted_matrix[1][0])).astype('uint8')

    return final_mask


def DICE_COE(mask_true, mask_prevision):
    # mask_prevision = 255 - mask_prevision #invert the result mask
    intersect = np.sum(mask_true*mask_prevision)
    fsum = np.sum(mask_true > 0)
    ssum = np.sum(mask_prevision > 0)
    dice = (2 * intersect) / (fsum + ssum)
    return dice

###################### pipeline ####################


# List of image paths
image_paths = [
    'allimages/ISIC_0000000.jpg',
    'allimages/ISIC_0000001.jpg',
    'allimages/ISIC_0000008.jpg',
    'allimages/ISIC_0000019.jpg',
    'allimages/ISIC_0000024.jpg',
    'allimages/ISIC_0000030.jpg',
    'allimages/ISIC_0000042.jpg',
    'allimages/ISIC_0000045.jpg',
    'allimages/ISIC_0000046.jpg',
    'allimages/ISIC_0000049.jpg',
    'allimages/ISIC_0000080.jpg',
    'allimages/ISIC_0000095.jpg',
    'allimages/ISIC_0000112.jpg',
    'allimages/ISIC_0000140.jpg',
    'allimages/ISIC_0000142.jpg',
    'allimages/ISIC_0000143.jpg',
    'allimages/ISIC_0000145.jpg',
    'allimages/ISIC_0000146.jpg',
    'allimages/ISIC_0000150.jpg',
    'allimages/ISIC_0000151.jpg',
]

dice = []

# Load each image and append to the images list
for i, path in enumerate(image_paths):
    img = cv2.imread(path)
    new_path = path.replace('.jpg', '_Segmentation.png')
    mask_true = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)

    colorspace = 'Gray'  # choose the colorspace to compare different colorspaces
    if (colorspace == 'LAB'):
        imLab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(imLab)
    if (colorspace == 'Gray'):
        l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    inpainted = remove_skin_hair(l, 40)

    good_region = find_good_region(l)

    mask_result = segmentation(inpainted, good_region)

    dice.append(DICE_COE(mask_true, mask_result))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))
    # Adjust the 'y' parameter for vertical position
    fig.suptitle("Image {:}".format(i), y=0.75)
    ax1.imshow(l, cmap='gray')
    ax1.set_title("Original image")
    ax1.set_axis_off()
    ax2.imshow(mask_result, cmap='gray')
    ax2.set_title("Computed mask, dice = {:.2f}".format(dice[i]))
    ax2.set_axis_off()
    ax3.imshow(mask_true, cmap='gray')
    ax3.set_title("True mask")
    ax3.set_axis_off()
    plt.show()

################ evaluation #################
dice = np.array(dice)
mean_dice = np.mean(dice[dice != 0])
std_dev_dice = np.std(dice[dice != 0], ddof=1)

image_dice = list(zip(range(20), dice))

print("{:} {:}".format("Mean dice", mean_dice))
print("{:} {:}".format("Standard Deviation", std_dev_dice))

print("-" * 30)

# Print the table header
print("\033[1m{:<15} {:<15}\033[0m".format("Image", "Dice"))

# Print a horizontal line
print("-" * 30)

# Define a function to color code the Dice coefficients


def color_code(value):
    if value >= 0.8:
        return "\033[92m{:<15}\033[0m".format(value)  # Green for values >= 0.8
    elif 0.7 <= value < 0.8:
        # Yellow for values in [0.7, 0.8)
        return "\033[93m{:<15}\033[0m".format(value)
    else:
        return "\033[91m{:<15}\033[0m".format(value)  # Red for values < 0.7


# Print the color-coded table rows
for row in image_dice:
    print("{:<15} {}".format(row[0], color_code(row[1])))
