import numpy as np
import cv2

# loading monochromatic image & converting BGR to RGB
path = '/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 1/imgs/monalisa.png'
image = cv2.imread(path)

# normalizing image to [0, 1]
image = (image.astype(np.float32)) / 255.0
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# taking R, G, B channels
R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]

# creating new channels
R_x = 0.393 * R + 0.769 * G + 0.189 * B
G_x = 0.349 * R + 0.686 * G + 0.168 * B
B_x = 0.272 * R + 0.534 * G + 0.131 * B

# clipping values to 0-1 range
R_x = np.clip(R_x, 0, 1)
G_x = np.clip(G_x, 0, 1)
B_x = np.clip(B_x, 0, 1)

# stacking channels back together
filtered_image_rgb = np.stack([R_x, G_x, B_x], axis=2)

# converting back to [0, 255]
filtered_image_rgb = np.array(filtered_image_rgb * 255, dtype=np.uint8)

# converting image back to BGR 
final_image = cv2.cvtColor(filtered_image_rgb, cv2.COLOR_RGB2BGR)

# showing image
cv2.imshow('Monalisa A', final_image)
cv2.imwrite('monalisa_a.png', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()