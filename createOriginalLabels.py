import cv2
import os
import csv
import pandas as pd
import logging


class LabelDrawer():
    def __init__(self, labels_path: str, image_folder: str, index: str):
        self.init_logger()
        self.labels_path = labels_path
        self.base_path = image_folder
        self.labels_df = pd.read_csv(labels_path)
        self.output_dir = os.path.join("E:\\Labeled", index)
        logging.info(f"labels: {self.labels_path}\nbase path: {self.base_path}\n"
                     f"output dir: {self.output_dir}")

        if not os.path.exists(self.output_dir):
            logging.info(f"Creating output dir {self.output_dir}")
            os.makedirs(self.output_dir)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.4
        self.thickness = 1

    def init_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def label_image(self, image_name: str):
        print(f"Labelling image: {image_name}")
        img = cv2.imread(os.path.join(self.base_path, image_name))
        filter = self.labels_df["filename"] == image_name
        matching_entries = self.labels_df.where(filter)
        matching_entries = matching_entries.dropna()

        for index, row in matching_entries.iterrows():
            x_min = int(row['xmin'])
            x_max = int(row['xmax'])
            y_min = int(row['ymin'])
            y_max = int(row['ymax'])
            label = row['class']

            if (label == 'stressed'):
                color = (0, 0, 0)
            else:
                color = (255, 0, 0)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                          color=color, thickness=2)
            img = cv2.putText(img, label, (x_min, y_min - 10),
                              self.font, self.fontScale, color, self.thickness, cv2.LINE_AA)

            cv2.imwrite(os.path.join(self.output_dir, image_name), img)
