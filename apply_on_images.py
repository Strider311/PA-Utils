import os
import logging
from ImageProcessingApproach import ImageProcessingClassifier
from createOriginalLabels import LabelDrawer
from draw_threshold import ThresholdPainter
import numpy as np

from test_cv_approach import AccuracyTester

root_dir = "C:\\Users\\saif_\\Main\\Projects\\PrecisionAgriculture\\data\\Processed\\Demo-04_12_2023-20_00_44\\ndvi"
rgb_dir = "C:\\Users\\saif_\\Main\\Projects\\PrecisionAgriculture\\data\\Input\\RGB_Images\\Train_Images"
labels_dir = "C:\\Users\\saif_\\Main\\Projects\\PrecisionAgriculture\\data\\Input\\Labels\\Train_Labels_CSV.csv"

numpy_dir = os.path.join(root_dir, "numpy")
ndvi_dir = os.path.join(root_dir, "images")

numpyClassifier = ImageProcessingClassifier(0.35, 0.6, numpy_dir)

accuracy_tester = AccuracyTester(
    labels_path=labels_dir, imageClassifier=numpyClassifier, processed_folder=numpy_dir, rgb_folder=rgb_dir)

accuracy_tester.process_image("Image_048.jpg")



