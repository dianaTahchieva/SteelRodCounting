
import cv2
import numpy as np

class SobelEdge:

    def get_sobel_edges_contour(slef, img):

        # Compute Sobel gradients
        sobel_h = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3)
        sobel_v = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)

        # Sobel derivatives basd on https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
        abs_grad_x= cv2.convertScaleAbs(sobel_h)
        abs_grad_y = cv2.convertScaleAbs(sobel_v)
        
        # Compute the magnitude of the gradients
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5,0)

        #converting back to CV_8U
        absx = cv2.convertScaleAbs(sobel_h, abs_grad_x)
        absy = cv2.convertScaleAbs(sobel_v, abs_grad_y)
        edges =  cv2.addWeighted(absx, 0.5, absy, 0.5, 0, grad)

        # Normalize the edges to the range [0, 255]
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)

        # Convert edges to uint8 format
        edges = np.uint8(edges)

        # Thresholding to obtain binary image
        _, binary_edges = cv2.threshold(edges, 25, 254, cv2.THRESH_BINARY)

        contours_sobel, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    
        # Filter contours based on size (remove small contours, consider large ones)
        filtered_contours_sobel = [cnt for cnt in contours_sobel if cv2.contourArea(cnt) > 100]

        return filtered_contours_sobel