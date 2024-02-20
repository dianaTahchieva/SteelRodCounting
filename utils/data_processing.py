
import cv2
import os
import pandas as pd
import numpy as np
import math
import tkinter as tk



class DataProcessing(object):

    """
    Resize image to a given aspect ration
    """
    def ResizeWithAspectRatio(self, image, area = 0.0, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if area != 0.0:
            vector = math.sqrt(area)
            root = tk.Tk()
            screen_h = root.winfo_screenheight()
            screen_w = root.winfo_screenwidth()

            window_h = screen_h * vector
            window_w = screen_w * vector

        if h > window_h or w > window_w:
            if h / window_h >= w / window_w:
                multiplier = window_h / h
        else:
            multiplier = window_w / w
        
            
        return cv2.resize(image, (0, 0), fx=multiplier, fy=multiplier)

    def extract_unique_data(self,filename):
        df = pd.read_csv(filename, header=None)
        unique_names = np.unique(df.iloc[:,0])
        return unique_names


