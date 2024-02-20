import cv2

class CannyEdge:

    def get_canny_edge_contour(self, image):
        # Apply edge detection using Canny
        edges = cv2.Canny(image, 30, 150, apertureSize=3)

        # Find contours in the image
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size (remove small contours, consider large ones)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

        # Count the number of contours (assumed to be steel bars)
        steel_bar_count = len(filtered_contours)
        return steel_bar_count