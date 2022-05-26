import matplotlib.pyplot as plt
import pydicom as dicom
import os
import pandas as pd
import numpy as np

db = pd.read_csv(r"C:/Users/aaron/Desktop/Coding/FPGA AI Project/mass_case_description_train_set.csv", skip_blank_lines=True)
db.dropna(axis=0, inplace=True, thresh=2)

for name in db[db.columns[11]].values:
    new_name = name.replace('000000.dcm', '1-1.dcm')
    db["image file path"].replace(name, new_name, regex=True, inplace=True)


# This is the path to the top directory containing the full images
full_image_dir = r"C:\Breast-Mass-Images\CBIS-DDSM\Mass-Training"


# Convert the dicom images to a numpy array
list_of_images = []



for name in db[db.columns[11]].values:
    image = dicom.dcmread(os.path.normpath(os.path.join(full_image_dir, name)))
    list_of_images.append(image.pixel_array)


db['Full_Images_Array'] = list_of_images


plt.imshow(db.iloc[0, 14], cmap=plt.cm.gray)
plt.show()
