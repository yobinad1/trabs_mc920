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

# bit lane extraction and showing 
for i in range(8):
    bit_plane = ((image >> i) & 1) * 255
    
    # showing images
    cv2.imshow(f'Plano de Bit {i}', bit_plane)
    cv2.imwrite(f'bit{i}.png', bit_plane)

cv2.waitKey(0)
cv2.destroyAllWindows()