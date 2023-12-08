import os.path

import numpy as np
import cv2

class ThresholdPainter():
    def __init__(self, base_image_folder: str, index: str, lower_bound: list, upper_bound: list):
        self.upper = np.array(upper_bound, dtype="uint8")
        self.lower = np.array(lower_bound, dtype="uint8")
        self.base_path = base_image_folder
        self.output_dir = os.path.join("/home/saif/WslHome/PrecisionAgri/data/threshold", index)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.4
        self.thickness = 1

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def draw_threshold(self, image_name: str):
        print(f"Drawing threshold on {image_name}")
        image = cv2.imread(os.path.join(self.base_path, image_name))
        original = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image, self.lower, self.upper)
        detected = cv2.bitwise_and(original, original, mask=mask)
        color = (255, 0, 0)

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours and find total area
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        area = 0
        for c in cnts:
            area += cv2.contourArea(c)
            cv2.putText(original, "Hello", (10, 10),  self.font, self.fontScale, color,
                        self.thickness, cv2.LINE_AA )
            cv2.drawContours(original, [c], 0, (0, 0, 0), 2)

        print(area)
        cv2.imshow(f'{image_name} - mask', mask)
        cv2.imshow(f'{image_name} - original', original)
        cv2.imshow(f'{image_name} - opening', opening)
        cv2.imshow(f'{image_name} - detected', detected)
        cv2.imwrite(os.path.join(self.output_dir, image_name), original)
        cv2.waitKey()


base_folder = "/home/saif/WslHome/PrecisionAgri/data/Processed/Testing-11_11_2023-21_11_03/ndvi/images"
upper = np.array([38, 255, 255], dtype="uint8")  
lower = np.array([0, 137, 0], dtype="uint8")
painter = ThresholdPainter(base_folder, "ndvi",  lower_bound=lower, upper_bound=upper)

painter.draw_threshold("Image_006.jpg")