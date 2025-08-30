import numpy as np
import cv2

# loading monochromatic image
path = '/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 1/imgs/baboon_monocromatica.png'
image = cv2.imread(path)

# hight and width of each partition
h, w = int(image.shape[0]/4), int(image.shape[1]/4)

# creating partitions
partitions = []

# 4x4 partitions of the image
for i in range(4):
    for j in range(4):
        partitions.append(image[i*h:(i+1)*h, j*w:(j+1)*w])
     
# reordering the partitions as asked   
partitions_reordered = [[partitions[5], partitions[10], partitions[12], partitions[2]],
                        [partitions[7], partitions[15], partitions[0], partitions[8]],
                        [partitions[11], partitions[13], partitions[1], partitions[6]],
                        [partitions[3], partitions[14], partitions[9], partitions[4]]]

# copy the original image to create the mosaic
mosaic = image.copy()

# creating the mosaic
for i in range(4):
    for j in range(4):
        mosaic[i*h:(i+1)*h, j*w:(j+1)*w] = partitions_reordered[i][j]

# showing the mosaic
cv2.imshow('Mosaic', mosaic)
cv2.imwrite('mosaic.png', mosaic)
cv2.waitKey(0)
cv2.destroyAllWindows()