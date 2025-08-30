import numpy as np
import cv2

# reading image
path = "/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 1/imgs/baboon_monocromatica.png"
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# filters
filters = [
    np.array([
        [0,  0, -1,  0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0,  0, -1,  0, 0]
    ]),
    (1/256) * np.array([
        [1,  4, 6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4, 6,  4, 1]
    ]),
    np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]),
    np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]),
    np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ]),
    (1/9) * np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]),
    np.array([
        [-1, -1, 2],
        [-1, 2, -1],
        [2, -1, -1]
    ]),
    np.array([
        [2, -1, -1],
        [-1, 2, -1],
        [-1, -1, 2]
    ]),
    (1/9) * np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]),
    (1/8) * np.array([
        [-1, -1, -1, -1, -1],
        [-1,  2,  2,  2, -1],
        [-1,  2,  8,  2, -1],
        [-1,  2,  2,  2, -1],
        [-1, -1, -1, -1, -1]
    ]),
    np.array([
        [-1, -1, 0],
        [-1, 0, 1], 
        [0, 1, 1]
    ])]

# creating new filter
new_filter = np.sqrt(np.power(filters[3], 2)+np.power(filters[2], 2))
filters.append(new_filter)

# converting image to [0,255]
min_val = np.min(image)
max_val = np.max(image)
image = (image - min_val) / (max_val - min_val)
image = (image * 255).astype(np.uint8)

# creating vector of convolutions
results = []
for filter in filters:
    conv = cv2.filter2D(image, -1, filter)
    results.append(conv)

# showing and saving images
for i in range(len(results)):
    cv2.imshow(f'Filter {i}', results[i])
    cv2.imwrite(f'filterh({i}).png', results[i])
cv2.waitKey(0)
cv2.destroyAllWindows()