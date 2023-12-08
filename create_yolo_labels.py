import os
import pandas as pd
from PIL import Image

base_dir = "C:\\Users\\saif_\\Main\\Projects\\PrecisionAgriculture\\data\\TrainingData\\RGB_Augmented\\Test_Images"
csv_input = "C:\\Users\\saif_\\Main\\Projects\\PrecisionAgriculture\\data\\TrainingData\\RGB_Augmented\\Test_Labels_CSV.csv"

images_dir = os.path.join(base_dir, "images")
output_labels_dir = os.path.join(base_dir, "labels")
images = os.listdir(images_dir)

csv_file = pd.read_csv(csv_input)

for img_name in images:
    filter = csv_file["filename"] == img_name
    matching_entries = csv_file.where(filter)
    matching_entries = matching_entries.dropna()

    img_path = os.path.join(base_dir, "images", img_name)
    img = Image.open(img_path)
    image_width = img.width
    image_height = img.height

    file_name = os.path.join(
        output_labels_dir, f"{img_name.replace('jpg', 'txt')}")
    f = open(file_name, "a")

    for index, row in matching_entries.iterrows():
        if (row["class"] == "stressed"):
            label = 0
        else:
            label = 1
        x_center = ((row["xmin"] + row["xmax"])/2)/image_width
        y_center = ((row["ymin"] + row["ymax"])/2)/image_height
        width = (row["xmax"] - row["xmin"])/image_width
        height = (row["ymax"] - row["ymin"])/image_height

        # class x_center y_center width heightm
        entry = f"{label} {x_center} {y_center} {width} {height}\n"
        f.write(entry)
        print(f"Writing: {entry} in file: {file_name}")
    f.close()
