import numpy as np
import cv2

# loading monochromatic image & converting BGR to RGB
path = '/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 1/imgs/monalisa.png'
image = cv2.imread(path)

# normalizing image to [0, 1]
image = (image.astype(np.float32)) / 255.0

# converting image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# taking R, G, B channels
R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]

# creating filter
I = 0.2989 * R + 0.5870 * G + 0.1140 * B

# clipping values to 0-1 range
I = np.clip(I, 0, 1)

# converting image back to BGR
I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)

# converting back to [0, 255]
I = np.array(I * 255, dtype=np.uint8)

# showing images
cv2.imshow('Monalisa B', I)
cv2.imwrite('monalisa_b.png', I)
cv2.waitKey(0)
cv2.destroyAllWindows()