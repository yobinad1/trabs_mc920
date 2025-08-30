import numpy as np
import matplotlib.pyplot as plt
import cv2

# loading watch image
path = "/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 1/imgs/watch.png"
image = cv2.imread(path)

# converting image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# applying gaussian blur
blurry_image = cv2.GaussianBlur(gray_image, (21,21), sigmaX=0, sigmaY=0)

# applying divide operation (sketch)
sketch = cv2.divide(gray_image, blurry_image, scale=256)

# showing images
cv2.imshow('Gray', gray_image)
cv2.imshow('Blurred and Gray', blurry_image)
cv2.imshow('Sktech', sketch)
cv2.imwrite('sketch.png', sketch)
cv2.imwrite('gray.png', gray_image)
cv2.imwrite('blurred.png', blurry_image)
cv2.waitKey(0)
cv2.destroyAllWindows()