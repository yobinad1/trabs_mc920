import numpy as np
import cv2

# loading monochromatic image & converting BGR to RGB
path = '/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 1/imgs/watch.png'
image = cv2.imread(path)
image = (image.astype(np.float32)) / 255.0
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# creating filter
filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

# applying filter
aged_image_rgb = cv2.transform(image_rgb, filter)

# clipping values to be in the range [0, 1]
aged_image_rgb = np.clip(aged_image_rgb, 0, 1)

# converting image back to BGR
aged_image_bgr = cv2.cvtColor(aged_image_rgb, cv2.COLOR_RGB2BGR)

# corverting back to [0, 255]
aged_image_bgr = np.array(aged_image_bgr * 255, dtype=np.ubyte)

# showing image
cv2.imshow('Image', aged_image_bgr)
cv2.imwrite('aged_image.png', aged_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()