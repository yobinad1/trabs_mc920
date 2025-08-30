import numpy as np
import cv2

# reading image
path = '/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 1/imgs/baboon_monocromatica.png'
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# converting image to [0,255]
min_val = np.min(image)
max_val = np.max(image)
image = (image - min_val) / (max_val - min_val)
image = (image * 255).astype(np.uint8)

# quantization function
def qtzation(image, level):
    step = 255 / (level - 1)
    qtz = np.round(image / step) * step
    return qtz.astype(np.uint8)

# leves of quantization
levels = [2, 4, 8, 16, 32, 64, 128, 256]

# showing and saving images
for level in levels:
    qtz = qtzation(image, level)
    cv2.imshow(f'{level} niveis', qtz)
    cv2.imwrite(f'qtz_{level}.png', qtz)
    
cv2.waitKey(0)
cv2.destroyAllWindows()