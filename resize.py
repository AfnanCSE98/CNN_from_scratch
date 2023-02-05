# write a function to resize all png images in training-a directory to 32x32 and save them in training-a-resized directory.
# The function should take the path to training-a directory as input and return the path to training-b directory as output.
# The function should also return the number of images in training-b directory.

import os
import sys 
import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_images(path):
    #load all images in the directory
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        if img is not None:
            images.append(img)
    #resize all images to 32x32
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, (180,180))
        resized_images.append(resized_image)
    #save all resized images in a new directory
    new_path = path + '-resized'
    os.mkdir(new_path)
    i = 0
    for filename in os.listdir(path):
        cv2.imwrite(os.path.join(new_path, filename), resized_images[i])
        i += 1
    
    return new_path, len(resized_images)

# take the first argument as the path to training-a directory
path = sys.argv[1]
# call the function
new_path, num_images = resize_images(path)
# print the path to training-b directory
print(new_path)
# print the number of images in training-b directory
print(num_images)