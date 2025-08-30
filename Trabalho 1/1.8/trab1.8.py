import numpy as np
import cv2

path = '/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 1/imgs/city.png'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# converting image to [0,255]
min_val = np.min(image)
max_val = np.max(image)
image = (image - min_val) / (max_val - min_val)
image = (image * 255).astype(np.uint8)

# applying convertion to [100,200]
image_transf = np.clip(image, 100, 200)
image_transf = cv2.equalizeHist(image_transf)

# negative
image_neg = - image + 255

# inversion of the even lines of the image
img_inv_even = image.copy()
img_inv_even[::2] = np.flip(img_inv_even[::2], axis=1)

# mirroring the superior half of the image
img_mirror_sup = image.copy()
img_mirror_sup[image.shape[0]//2:,:] = np.flip(image[0:image.shape[0]//2,:], axis=0)

# vertical mirroring
img_mirror_inf = image.copy()
img_mirror_inf= np.flip(image, axis=0)

# showing images
cv2.imshow('Original', image)
cv2.imshow('Negative', image_neg)
cv2.imshow('Transformed', image_transf)
cv2.imshow('Even lines', img_inv_even)
cv2.imshow('Mirror superior', img_mirror_sup)
cv2.imshow('Mirror inferior', img_mirror_inf)
cv2.imwrite('img_neg.png', image_neg)
cv2.imwrite('img_inv_even.png', img_inv_even)
cv2.imwrite('img_mirror_sup.png', img_mirror_sup)
cv2.imwrite('img_mirror_inf.png', img_mirror_inf)
cv2.imwrite('img_transf.png', image_transf)

cv2.waitKey(0)
cv2.destroyAllWindows()
