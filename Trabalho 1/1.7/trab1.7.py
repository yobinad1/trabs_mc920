import numpy as np
import cv2

# loading monochromatic image
path1 = '/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 1/imgs/baboon_monocromatica.png'
image1 = cv2.imread(path1)

# converting image to [0,255]
min_val = np.min(image1)
max_val = np.max(image1)
image1 = (image1 - min_val) / (max_val - min_val)
image1 = (image1 * 255).astype(np.uint8)

# normalizing image to [0,1]
image1 = (image1.astype(np.float32)) / 255.0

# loading monochromatic image & converting BGR to RGB
path2 = '/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 1/imgs/butterfly.png'
image2 = cv2.imread(path2)

# converting image to [0,255]
min_val = np.min(image2)
max_val = np.max(image2)
image2 = (image2 - min_val) / (max_val - min_val)
image2 = (image2 * 255).astype(np.uint8)

# normalizing image to [0,1]
image2 = (image2.astype(np.float32)) / 255.0

# mixing images
image3 = 0.2*image1 + 0.8*image2
image4 = 0.5*image1 + 0.5*image2
image5 = 0.8*image1 + 0.2*image2

# converting back to [0,255]
image3 = np.array(image3 * 255, dtype=np.ubyte)
image4 = np.array(image4 * 255, dtype=np.ubyte)
image5 = np.array(image5 * 255, dtype=np.ubyte)

# showing images
cv2.imshow('0.2A', image3)
cv2.imshow('0.5A', image4)
cv2.imshow('0.8A', image5)
cv2.imwrite('0.2A.png', image3)
cv2.imwrite('0.5A.png', image4)
cv2.imwrite('0.8A.png', image5)
cv2.waitKey(0)
cv2.destroyAllWindows()