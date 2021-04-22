# convert files into proper valid jpg
import os
import shutil
from PIL import Image

folder = "screenshots"
path = "raw_data/stupa_screenshots"  # Source Folder
dest_folder = "raw_data/" + str(folder)
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

    if path[-1] != '/':
        path = path + '/'
    for file in os.listdir(path):
        try:
            extension = file.split('.')[-1]
            name = file.split('.')[0] + '.jpg'
            fileLoc = path + file
            img = Image.open(fileLoc)
            new = Image.new("RGB", img.size, (255, 255, 255))
            new.paste(img, None)  # save the new image with jpg as extension
            new.save(dest_folder + "/" + name, 'JPEG', quality=100)
        except Exception as e:
            pass
    shutil.rmtree(path)
########################################################################################################################
# Excel to csv and train test split with respective csv

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel("raw_data/annotation.xlsx")

train, test = train_test_split(df, test_size=0.2, shuffle=True)
train_path = train['name'].tolist()
test_path = test['name'].tolist()

shutil.rmtree("images")
os.makedirs("images")
os.makedirs("images/train")
os.makedirs("images/test")
os.makedirs("images/results")

try:
    for imageName in train_path:
        imageName = "".join(imageName.split())
        shutil.copy(os.path.join(dest_folder, imageName), "images/train")
    for imageName in test_path:
        imageName = "".join(imageName.split())
        shutil.copy(os.path.join(dest_folder, imageName), "images/test")
except Exception as e:
    print("bc")


train.to_csv("images/train/train.csv", index=False)
test.to_csv("images/test/test.csv", index=False)
