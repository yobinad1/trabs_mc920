import numpy as np
import cv2
import os

# checks if pixel is valid
def is_valid_pixel(h, w, row, col):
    return 0 <= row < h and 0 <= col < w

def distribute_err(img, row, col, ch, err, mask, half_width):
    h, w = img.shape[:2]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            r, c = row + i, col - half_width + j
            if is_valid_pixel(h, w, r, c):
                img[r, c, ch] = np.clip(img[r, c, ch] + err * mask[i, j], 0, 255)

def threshold(pixel_value):
    return 0 if pixel_value < 128 else 1

def meio_tom(image, mask):
    img = image.copy().astype(np.int32)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    
    output_img = np.zeros_like(img)
    half_width = mask.shape[1] // 2

    h, w, channels = img.shape
    for ch in range(channels):
        for row in range(h):
            for col in range(w):
                output_img[row, col, ch] = threshold(img[row, col, ch])
                err = img[row, col, ch] - output_img[row, col, ch] * 255
                distribute_err(img, row, col, ch, err, mask, half_width)
    
    return output_img * 255

def meio_tom_varredura_variada(image, mask):
    img = image.copy().astype(np.int32)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    
    output_img = np.zeros_like(img)
    mask_flip = np.fliplr(mask)
    half_width = mask.shape[1] // 2

    h, w, channels = img.shape
    for ch in range(channels):
        for row in range(h):
            if row % 2 == 0:
                cols = range(w)
                current_mask = mask
            else:
                cols = range(w-1, -1, -1)
                current_mask = mask_flip

            for col in cols:
                output_img[row, col, ch] = threshold(img[row, col, ch])
                err = img[row, col, ch] - output_img[row, col, ch] * 255
                distribute_err(img, row, col, ch, err, current_mask, half_width)
    
    return output_img * 255


dist_err = {
    'floyd&steinberg': np.array([
            [0   ,   0   ,   7/16],
            [3/16,  5/16 ,   1/16]
        ]),
    'stevenson&arce': np.array([
            [  0   ,   0   ,    0    ,   0    ,   0    , 32/200 ,   0   ],
            [12/200,   0   ,  26/200 ,   0    , 30/200 ,    0   , 16/200],
            [  0   , 12/200,    0    , 26/200 ,   0    , 12/200 ,   0   ],
            [ 5/200,   0   ,  12/200 ,   0    , 12/200 ,    0   ,  5/200]
        ]),
    'burkes': np.array([
            [  0   ,   0   ,    0    ,   8/32  ,   4/32  ],
            [ 2/32 , 4/32  ,  8/32   ,   4/32  ,   2/32  ]
        ]),
    'sierra': np.array([
            [  0   ,   0   ,    0    ,   5/32  ,   3/32  ],
            [ 2/32 , 4/32  ,  5/32   ,   4/32  ,   2/32  ],
            [  0   , 2/32  ,  3/32   ,   2/32  ,    0    ]
        ]),
    'stucki': np.array([
            [  0   ,   0   ,    0    ,   8/42  ,   4/42  ],
            [ 2/42 , 4/42  ,  8/42   ,   4/42  ,   2/42  ],
            [ 1/42 , 2/42  ,  4/42   ,   2/42  ,   1/42  ]
        ]),
    'j&j&n': np.array([
            [  0   ,   0   ,    0    ,   7/48  ,   5/48  ],
            [ 3/48 , 5/48  ,  7/48   ,   5/48  ,   3/48  ],
            [ 1/48 , 3/48  ,  5/48   ,   3/48  ,   1/48  ]
        ])
}

# path to save the output images
path_out = '/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 2/item1'

images = {
    'baboon_mono': cv2.imread("/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 2/imgs/baboon_monocromatica.png"),
    'baboon_color': cv2.imread("/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 2/imgs/baboon_colorida.png"),
    'butterfly': cv2.imread("/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 2/imgs/butterfly.png")
}

for img_name, img in images.items():
    for mask_name, mask in dist_err.items():
        output1 = meio_tom(img, mask)
        output2 = meio_tom_varredura_variada(img, mask)
        
        # salva as imagens
        filename1 = f"{img_name}_{mask_name}_meiotom.png"
        filename2 = f"{img_name}_{mask_name}_meiotom_varredura.png"
        
        cv2.imwrite(os.path.join(path_out, filename1), output1.astype(np.uint8))
        cv2.imwrite(os.path.join(path_out, filename2), output2.astype(np.uint8))