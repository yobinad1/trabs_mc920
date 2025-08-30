import numpy as np
import cv2

# loading monochromatic image
path = '/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 1/imgs/baboon_monocromatica.png'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# converting image to [0,255]
min_val = np.min(image)
max_val = np.max(image)
image = (image - min_val) / (max_val - min_val)
image = (image * 255).astype(np.uint8)

# normalizing image
image = image / 255

# gammas
gamma_1 = 1.5
gamma_2 = 2.5
gamma_3 = 3.5

# applying gamma correction
beta1 = np.power(image, 1/gamma_1)
beta2 = np.power(image, 1/gamma_2)
beta3 = np.power(image, 1/gamma_3)

# converting back to [0,255]
beta1 = np.array(beta1*255, dtype=np.ubyte)
beta2 = np.array(beta2*255, dtype=np.ubyte)
beta3 = np.array(beta3*255, dtype=np.ubyte)

# showing image
cv2.imshow('Gamma 1.5', beta1)
cv2.imshow('Gamma 2.5', beta2)
cv2.imshow('Gamma 3.5', beta3)
cv2.imwrite('gamma_1.5.png', beta1)
cv2.imwrite('gamma_2.5.png', beta2)
cv2.imwrite('gamma_3.5.png', beta3)
cv2.waitKey(0)
cv2.destroyAllWindows()