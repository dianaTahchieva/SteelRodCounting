import cv2
import numpy as np

class ImageSegmentation:

    def image_segmentation(self, image):

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply median Blur
        blur = cv2.medianBlur(gray,5) 

        # Adaptive thresholding showed more reliable results when applied to test set
        threshold = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
        imgDial = cv2.dilate(threshold, (5, 5), iterations=3)
        masked_img= cv2.erode(imgDial, np.ones((7, 7), np.uint8))

        return masked_img
        

        # Compare the results and choose the count
        counts = [num_components_1, num_components_2, num_components_3]
        #chosen_count = max(set(counts), key=counts.count)  # Choose the most common count

        return counts
    
    def get_label_contours(self, image):
        # Count connected components or blobs        
        _, labels = cv2.connectedComponents(image)
        num_components = labels.max()
        return num_components
    
    def get_marker_contours(self, image):
        # Marker-based segmentation
        markers = cv2.connectedComponents(image)
        num_components = markers[0]
        return num_components
    
    def get_edge_contours(self, image):
        # Find contours in the binary edges
        #RETR_EXTERNAL  - retrieves only the extreme outer contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size (remove small contours, consider large ones)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        num_components = len(filtered_contours)
        return num_components