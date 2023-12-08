from ImageProcessingApproach import ImageProcessingClassifier
import os
import cv2
import pandas as pd


class ConfusionMatrix():
    def __init__(self) -> None:
        self.falseHealthy = 0
        self.trueHealthy = 0
        self.falseUnhealthy = 0
        self.trueUnhealthy = 0


class AccuracyTester():
    def __init__(self, labels_path: str, rgb_folder: str, processed_folder: str, imageClassifier: ImageProcessingClassifier, confusionMatrix: ConfusionMatrix):
        self.__init_cv2_params__()
        self.image_processor = imageClassifier
        self.labels_path = labels_path
        self.processed_folder = processed_folder
        self.rgb_folder = rgb_folder
        self.labels_df = pd.read_csv(labels_path)
        self.confusionMatrix = confusionMatrix

    def __init_cv2_params__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.4
        self.thickness = 1

    def process_image(self, image_name: str):
        print(f"Labelling image: {image_name}")
        processed_file_name = os.path.join(
            self.processed_folder, image_name.replace("jpg", "txt"))
        filter = self.labels_df["filename"] == image_name
        matching_entries = self.labels_df.where(filter)
        matching_entries = matching_entries.dropna()
        rows = matching_entries.iterrows()
        for index, row in rows:
            i = 1
            x_min = int(row['xmin'])
            x_max = int(row['xmax'])
            y_min = int(row['ymin'])
            y_max = int(row['ymax'])
            label = row['class']

            healthy_img, healthy_area = self.image_processor.extract_healthy_from_subimage(
                processed_file_name, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

            unhealthy_img, unhealthy_area = self.image_processor.extract_unhealthy_from_subimage(
                processed_file_name, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

            if (healthy_area == 0 or unhealthy_area == 0):
                continue

            if (unhealthy_area > healthy_area):
                processed_label = "stressed"
            elif (healthy_area >= unhealthy_area):
                processed_label = "healthy"

            if (label == "stressed" and processed_label == "stressed"):
                self.confusionMatrix.trueUnhealthy += 1
            elif (label == "stressed" and processed_label == "healthy"):
                self.confusionMatrix.falseHealthy += 1
            elif (label == "healthy" and processed_label == "healthy"):
                self.confusionMatrix.trueHealthy += 1
            elif (label == "healthy" and processed_label == "stressed"):
                self.confusionMatrix.falseUnhealthy += 1

            print(f"--------{image_name} Box #{i}---------")
            print(f"Filename: {processed_file_name}\n")
            print(
                f"Healthy area %: {healthy_area/(healthy_area+unhealthy_area)}")
            print(
                f"Unhealthy area %: {unhealthy_area/(healthy_area + unhealthy_area)}")
            print(f"Original label: {label}")
            print(F"Processed label: {processed_label}")
            print(f"-----------------------------------------")


root_dir = "C:\\Users\\saif_\\Main\\Projects\\PrecisionAgriculture\\data\\Processed\\Demo-05_12_2023-20_13_29\\ndvi"
rgb_dir = "C:\\Users\\saif_\\Main\\Projects\\PrecisionAgriculture\\data\\Input\\RGB_Images\\Train_Images"
labels_dir = "C:\\Users\\saif_\\Main\\Projects\\PrecisionAgriculture\\data\\Input\\Labels\\Train_Labels_CSV.csv"

numpy_dir = os.path.join(root_dir, "numpy")
ndvi_dir = os.path.join(root_dir, "images")

numpyClassifier = ImageProcessingClassifier(0.1, 0.5, numpy_dir)
confusionMatrix = ConfusionMatrix()
accuracy_tester = AccuracyTester(
    labels_path=labels_dir, imageClassifier=numpyClassifier, processed_folder=numpy_dir, rgb_folder=rgb_dir, confusionMatrix=confusionMatrix)


filesToProcess = os.listdir(rgb_dir)

for file in filesToProcess:
    accuracy_tester.process_image(file)

print(f'''RESULT:
        TRUE HEALTHY: {confusionMatrix.trueHealthy}\n
        FALSE HEALTHY: {confusionMatrix.falseHealthy}\n
        TRUE STRESSED: {confusionMatrix.trueUnhealthy}\n
        FALSE STRESSED: {confusionMatrix.falseUnhealthy}''')

print(
    f"healthy normalized: {confusionMatrix.trueHealthy/(confusionMatrix.trueHealthy+confusionMatrix.falseHealthy)}")
print(
    f"stressed normalized: {confusionMatrix.trueUnhealthy/(confusionMatrix.trueUnhealthy+confusionMatrix.falseUnhealthy)}")
