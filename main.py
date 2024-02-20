import cv2
import numpy as np
import os

from algorithms import  image_segmentation_threshold, sobel_edge_detection, canny_edge_detection
from utils.data_processing import DataProcessing



def count_steel_bars_canny(image):
    
    canny = canny_edge_detection.CannyEdge()
            
    # Convert the image to grayscale/HSV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(gray,5)
    
    steel_bar_count = canny.get_canny_edge_contour(blur)
    return steel_bar_count



def count_steel_bars_sobel_edge(img):

    sobel = sobel_edge_detection.SobelEdge()

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 10)

    filtered_contours_sobel = sobel.get_sobel_edges_contour(blur)    

    return len(filtered_contours_sobel)

def count_steel_bars_image_segmentation(img):

    im_segm = image_segmentation_threshold.ImageSegmentation()

    masked_img = im_segm.image_segmentation(img)

    num_components_label = im_segm.get_label_contours(masked_img)
        
    num_components_marker = im_segm.get_marker_contours(masked_img)

    num_components_contour = im_segm.get_edge_contours(masked_img)

    return num_components_label, num_components_marker, num_components_contour


def weighted_voting(counts, weights):
    total_weight = sum(weights.values())
    if total_weight == 0:
        return None  # All weights are zero

    weighted_sum = sum(counts[method] * weights[method] for method in counts.keys())
    final_count = weighted_sum / total_weight
    return round(final_count)


def count_steel_bars_in_folder(folder_path, test_set_images):
    total_steel_bars = 0
    
    # Iterate through all images in the folder
    for filename in os.listdir(folder_path):
        if filename in test_set_images:

            image_path = os.path.join(folder_path, filename)
            # Read the image
            image = cv2.imread(image_path)
           
            steel_bar_count_canny = count_steel_bars_canny(image)
            steel_bar_count_sobel = count_steel_bars_sobel_edge(image)
            num_components_label, num_components_marker, num_components_contour = count_steel_bars_image_segmentation(image)

            print(f"Image: {filename}, Steel Bars Count sobel: {steel_bar_count_sobel},\
                   Steel Bars Count image segment label: {num_components_label}, Steel Bars Count image segment marker: {num_components_marker},\
                     Steel Bars Count image segment contour: {num_components_contour}, Steel Bars Count Canny:{steel_bar_count_canny}")
            
            # Compare the results and choose the count
            counts = [steel_bar_count_sobel, num_components_label, num_components_marker, num_components_contour, steel_bar_count_canny]

            #majority voting mechanism
            chosen_count =  max(set(counts), key=counts.count)  # Choose the most common count
            print(f"Image: {filename}, Steel Bars Count based on majority voting: {chosen_count} \n")

            #total_steel_bars += steel_bar_count
            # Alternatively we can use weighted voting (assign weights based on confidence)
            counts_dict = {'canny': steel_bar_count_canny, 'sobel':steel_bar_count_sobel, \
                           'label': num_components_label, "marker": num_components_marker, "contour": num_components_contour, } 
            weights = {'canny': 0.2, 'sobel': 0.5, 'label': 0.3, "marker":0.3, "contour":0.2}

            final_count_weighted = weighted_voting(counts_dict, weights)
            print(f"Image: {filename}, Steel Bars Count based on weighted count: {final_count_weighted} \n")
    
    print(f"Total Steel Bars in the Image Database: {total_steel_bars}")


def main():
    dp = DataProcessing()
    path = 'C:/Users/nickj/OneDrive/Documents/vs_workspace/github/SteelRodCounting/data/RebarDSC/images/'
    test_set_images = dp.extract_unique_data("C:/Users/nickj/OneDrive/Documents/vs_workspace/github/SteelRodCounting/data/RebarDSC/annotations/test.csv")
    
    count_steel_bars_in_folder(path, test_set_images)


if __name__ == "__main__":
    main()

