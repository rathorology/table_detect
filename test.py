import os
import shutil

import cv2

data_directory = "dobby Latest Screenshots"

count = 1
new_count = 155
for c in range(0, len(os.listdir(data_directory))):
    shutil.copy(os.path.join(data_directory, str(count) + ".jpg"), "new/" + str(new_count) + ".jpg")
    count += 1
    new_count += 1
