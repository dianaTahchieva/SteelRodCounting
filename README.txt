# Object Counting: Steel Bars in RebarDSC

## Overview
This repository contains code for counting steel bars in images from the RebarDSC dataset using various unsupervised methods, including edge detection and image segmentation via thresholding.

## Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Folder Structure](#folder-structure)
- [Setup](#setup)


## Introduction
The goal of this project is to count steel bars in images without relying on provided labels. The code employs a combination of edge detection and image segmentation techniques to achieve accurate counting across diverse scenarios.

## Methodology
The methodologies include Sobel and Canny edge detection, image segmentation via thresholding, and an ensemble approach to leverage the strengths of multiple methods. Various contour extraction and counting algorithms are explored to improve accuracy.

## Folder Structure
- `data/`: Contains input images and dataset-related files.
- `algorithms/`: Includes implementation of edge detection, image segmentation, and ensemble approaches.
- `results/`: Stores output files and visualizations.
- `README.txt`: Overview of the project, setup, and usage instructions.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/dianaTahchieva/SteelRodCounting.git
   
