import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def scale_image(image, scale_factor, interpolation_method='nearest'):
    h, w = image.shape[:2]
    
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    output = np.zeros((new_h, new_w, 3), dtype=np.uint8) if len(image.shape) == 3 else np.zeros((new_h, new_w), dtype=np.uint8)
    
    for y_out in range(new_h):
        for x_out in range(new_w):
            x_in = x_out / scale_factor
            y_in = y_out / scale_factor
            
            if interpolation_method == "nearest":
                x_nearest = round(x_in)
                y_nearest = round(y_in)
                
                x_nearest = min(max(0, x_nearest), w - 1)
                y_nearest = min(max(0, y_nearest), h - 1)
                
                output[y_out, x_out] = image[y_nearest, x_nearest]
            
            elif interpolation_method == "bilinear":
                x_floor = int(np.floor(x_in))
                y_floor = int(np.floor(y_in))
                x_ceil = min(x_floor + 1, w - 1)
                y_ceil = min(y_floor + 1, h - 1)

                dx = x_in - x_floor
                dy = y_in - y_floor

                value = (1-dx)*(1-dy)*image[y_floor, x_floor] + \
                        dx*(1-dy)*image[y_floor, x_ceil] + \
                        (1-dx)*dy*image[y_ceil, x_floor] + \
                        dx*dy*image[y_ceil, x_ceil]

                output[y_out, x_out] = value.astype(np.uint8) if value.dtype != np.uint8 else value
            
            elif interpolation_method == "bicubic":
                if 2 <= x_in < w-2 and 2 <= y_in < h-2:
                    output[y_out, x_out] = bicubic_interpolation(image, x_in, y_in)
                else:
                    x_floor = int(np.floor(x_in))
                    y_floor = int(np.floor(y_in))
                    x_ceil = min(x_floor + 1, w - 1)
                    y_ceil = min(y_floor + 1, h - 1)

                    dx = x_in - x_floor
                    dy = y_in - y_floor

                    value = (1-dx)*(1-dy)*image[y_floor, x_floor] + \
                            dx*(1-dy)*image[y_floor, x_ceil] + \
                            (1-dx)*dy*image[y_ceil, x_floor] + \
                            dx*dy*image[y_ceil, x_ceil]

                    output[y_out, x_out] = value.astype(np.uint8) if value.dtype != np.uint8 else value
            
            elif interpolation_method == "lagrange":
                if 2 <= x_in < w-2 and 2 <= y_in < h-2:
                    output[y_out, x_out] = lagrange_interpolation(image, x_in, y_in)
                else:
                    x_floor = int(np.floor(x_in))
                    y_floor = int(np.floor(y_in))
                    x_ceil = min(x_floor + 1, w - 1)
                    y_ceil = min(y_floor + 1, h - 1)

                    dx = x_in - x_floor
                    dy = y_in - y_floor

                    value = (1-dx)*(1-dy)*image[y_floor, x_floor] + \
                            dx*(1-dy)*image[y_floor, x_ceil] + \
                            (1-dx)*dy*image[y_ceil, x_floor] + \
                            dx*dy*image[y_ceil, x_ceil]

                    output[y_out, x_out] = value.astype(np.uint8) if value.dtype != np.uint8 else value
    
    return output

def rotate_image(image, angle_degrees, interpolation_method="nearest"):
    h, w = image.shape[:2]
    
    angle_rad = np.radians(angle_degrees)
    
    output = np.zeros_like(image)

    center_x, center_y = w / 2, h / 2
    
    for y_out in range(h):
        for x_out in range(w):
            x_centered = x_out - center_x
            y_centered = y_out - center_y
            
            x_rotated = x_centered * np.cos(angle_rad) + y_centered * np.sin(angle_rad)
            y_rotated = -x_centered * np.sin(angle_rad) + y_centered * np.cos(angle_rad)
            
            x_in = x_rotated + center_x
            y_in = y_rotated + center_y
            
            if 0 <= x_in < w and 0 <= y_in < h:
                if interpolation_method == "nearest":
                    x_nearest = round(x_in)
                    y_nearest = round(y_in)

                    x_nearest = min(max(0, x_nearest), w - 1)
                    y_nearest = min(max(0, y_nearest), h - 1)
    
                    output[y_out, x_out] = image[y_nearest, x_nearest]
                
                elif interpolation_method == "bilinear":
                    x_floor = int(np.floor(x_in))
                    y_floor = int(np.floor(y_in))
                    x_ceil = min(x_floor + 1, w - 1)
                    y_ceil = min(y_floor + 1, h - 1)

                    dx = x_in - x_floor
                    dy = y_in - y_floor
                   
                    value = (1-dx)*(1-dy)*image[y_floor, x_floor] + \
                            dx*(1-dy)*image[y_floor, x_ceil] + \
                            (1-dx)*dy*image[y_ceil, x_floor] + \
                            dx*dy*image[y_ceil, x_ceil]

                    output[y_out, x_out] = value.astype(np.uint8) if value.dtype != np.uint8 else value
                
                elif interpolation_method == "bicubic":
                    if 2 <= x_in < w-2 and 2 <= y_in < h-2:
                        output[y_out, x_out] = bicubic_interpolation(image, x_in, y_in)
                    else:
                        x_floor = int(np.floor(x_in))
                        y_floor = int(np.floor(y_in))
                        x_ceil = min(x_floor + 1, w - 1)
                        y_ceil = min(y_floor + 1, h - 1)

                        dx = x_in - x_floor
                        dy = y_in - y_floor
                       
                        value = (1-dx)*(1-dy)*image[y_floor, x_floor] + \
                                dx*(1-dy)*image[y_floor, x_ceil] + \
                                (1-dx)*dy*image[y_ceil, x_floor] + \
                                dx*dy*image[y_ceil, x_ceil]

                        output[y_out, x_out] = value.astype(np.uint8) if value.dtype != np.uint8 else value
    
                elif interpolation_method == "lagrange":
                    if 2 <= x_in < w-2 and 2 <= y_in < h-2:
                        output[y_out, x_out] = lagrange_interpolation(image, x_in, y_in)
                    else:
                        x_floor = int(np.floor(x_in))
                        y_floor = int(np.floor(y_in))
                        x_ceil = min(x_floor + 1, w - 1)
                        y_ceil = min(y_floor + 1, h - 1)

                        dx = x_in - x_floor
                        dy = y_in - y_floor
                       
                        value = (1-dx)*(1-dy)*image[y_floor, x_floor] + \
                                dx*(1-dy)*image[y_floor, x_ceil] + \
                                (1-dx)*dy*image[y_ceil, x_floor] + \
                                dx*dy*image[y_ceil, x_ceil]

                        output[y_out, x_out] = value.astype(np.uint8) if value.dtype != np.uint8 else value
    return output

def P(t):
    return t if t > 0 else 0

def R(s):
    return (1/6) * (P(s+2)**3 - 4*P(s+1)**3 + 6*P(s)**3 - 4*P(s-1)**3)

def bicubic_interpolation(image, x, y):
    h, w = image.shape[:2]
    
    x_base = int(np.floor(x)) - 1
    y_base = int(np.floor(y)) - 1
    
    dx = x - np.floor(x)
    dy = y - np.floor(y)
    
    result = np.zeros_like(image[0, 0], dtype=np.float64)
    
    for m in range(-1, 3):
        for n in range(-1, 3):
            xi = x_base + m
            yj = y_base + n
            
            if 0 <= xi < w and 0 <= yj < h:
                weight = R(m - dx) * R(dy - n)
                pixel_value = image[yj, xi].astype(np.float64)
                result += weight * pixel_value
    
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

def lagrange_interpolation(image, x, y):
    h, w = image.shape[:2]
    
    x_base = int(np.floor(x)) - 1
    y_base = int(np.floor(y)) - 1
    
    dx = x - np.floor(x)
    dy = y - np.floor(y)
    
    result = np.zeros_like(image[0, 0], dtype=np.float64)
    
    def L(n):
        L_value = np.zeros_like(image[0, 0], dtype=np.float64)
        
        xi = x_base
        yj = y_base + (n - 2)
        if 0 <= xi < w and 0 <= yj < h:
            coef = -dx * (dx - 1) * (dx - 2) / 6
            L_value += coef * image[yj, xi].astype(np.float64)
        
        xi = x_base + 1
        if 0 <= xi < w and 0 <= yj < h:
            coef = (dx + 1) * (dx - 1) * (dx - 2) / 2
            L_value += coef * image[yj, xi].astype(np.float64)
        
        xi = x_base + 2
        if 0 <= xi < w and 0 <= yj < h:
            coef = -dx * (dx + 1) * (dx - 2) / 2
            L_value += coef * image[yj, xi].astype(np.float64)
        
        xi = x_base + 3
        if 0 <= xi < w and 0 <= yj < h:
            coef = dx * (dx + 1) * (dx - 1) / 6
            L_value += coef * image[yj, xi].astype(np.float64)
        
        return L_value
    
    
    coef = -dy * (dy - 1) * (dy - 2) / 6
    result += coef * L(1)
    
    coef = (dy + 1) * (dy - 1) * (dy - 2) / 2
    result += coef * L(2)
    
    coef = -dy * (dy + 1) * (dy - 2) / 2
    result += coef * L(3)
    
    coef = dy * (dy + 1) * (dy - 1) / 6
    result += coef * L(4)
    
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

path_in = "/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 4/imgs"
path_out = "/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 4/pt2"

out_butterfly = os.path.join(path_out, "out_butterfly")
out_monalisa = os.path.join(path_out, "out_monalisa")
os.makedirs(out_butterfly, exist_ok=True)
os.makedirs(out_monalisa, exist_ok=True)

image = cv2.imread(f"{path_in}/monalisa.png")
image2 = cv2.imread(f"{path_in}/butterfly.png")

## scale butterfly
scaled_butterfly_nearest = scale_image(image2, 2.5, interpolation_method="nearest")
scaled_butterfly_bilinear = scale_image(image2, 2.5, interpolation_method="bilinear")
scaled_butterfly_bicubic = scale_image(image2, 2.5, interpolation_method="bicubic")
scaled_butterfly_lagrange = scale_image(image2, 2.5, interpolation_method="lagrange")

# rotate butterfly
rotated_butterfly_nearest = rotate_image(image2, 22.5, interpolation_method="nearest")
rotated_butterfly_bilinear = rotate_image(image2, 22.5, interpolation_method="bilinear")
rotated_butterfly_bicubic = rotate_image(image2, 22.5, interpolation_method="bicubic")
rotated_butterfly_lagrange = rotate_image(image2, 22.5, interpolation_method="lagrange")

cv2.imwrite(f"{path_out}/scaled_butterfly_nearest.png", scaled_butterfly_nearest)
cv2.imwrite(f"{path_out}/scaled_butterfly_bilinear.png", scaled_butterfly_bilinear)
cv2.imwrite(f"{path_out}/scaled_butterfly_bicubic.png", scaled_butterfly_bicubic)
cv2.imwrite(f"{path_out}/scaled_butterfly_lagrange.png", scaled_butterfly_lagrange)
cv2.imwrite(f"{path_out}/rotated_butterfly_nearest.png", rotated_butterfly_nearest)
cv2.imwrite(f"{path_out}/rotated_butterfly_bilinear.png", rotated_butterfly_bilinear)
cv2.imwrite(f"{path_out}/rotated_butterfly_bicubic.png", rotated_butterfly_bicubic)
cv2.imwrite(f"{path_out}/rotated_butterfly_lagrange.png", rotated_butterfly_lagrange)

# scale baboon
scaled_monalisa_nearest = scale_image(image, 2.5, interpolation_method="nearest")
scaled_monalisa_bilinear = scale_image(image, 2.5, interpolation_method="bilinear")
scaled_monalisa_bicubic = scale_image(image, 2.5, interpolation_method="bicubic")
scaled_monalisa_lagrange = scale_image(image, 2.5, interpolation_method="lagrange")

# rotate baboon
rotated_monalisa_nearest = rotate_image(image, 22.5, interpolation_method="nearest")
rotated_monalisa_bilinear = rotate_image(image, 22.5, interpolation_method="bilinear")
rotated_monalisa_bicubic = rotate_image(image, 22.5, interpolation_method="bicubic")
rotated_monalisa_lagrange = rotate_image(image, 22.5, interpolation_method="lagrange")

cv2.imwrite(f"{path_out}/scaled_monalisa_nearest.png", scaled_monalisa_nearest)
cv2.imwrite(f"{path_out}/scaled_monalisa_bilinear.png", scaled_monalisa_bilinear)
cv2.imwrite(f"{path_out}/scaled_monalisa_bicubic.png", scaled_monalisa_bicubic)
cv2.imwrite(f"{path_out}/scaled_monalisa_lagrange.png", scaled_monalisa_lagrange)
cv2.imwrite(f"{path_out}/rotated_monalisa_nearest.png", rotated_monalisa_nearest)
cv2.imwrite(f"{path_out}/rotated_monalisa_bilinear.png", rotated_monalisa_bilinear)
cv2.imwrite(f"{path_out}/rotated_monalisa_bicubic.png", rotated_monalisa_bicubic)
cv2.imwrite(f"{path_out}/rotated_monalisa_lagrange.png", rotated_monalisa_lagrange)

# converting images to RGB for matplotlib
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
scaled_monalisa_nearest = cv2.cvtColor(scaled_monalisa_nearest, cv2.COLOR_BGR2RGB)
scaled_monalisa_bilinear = cv2.cvtColor(scaled_monalisa_bilinear, cv2.COLOR_BGR2RGB)
scaled_monalisa_bicubic = cv2.cvtColor(scaled_monalisa_bicubic, cv2.COLOR_BGR2RGB)
scaled_monalisa_lagrange = cv2.cvtColor(scaled_monalisa_lagrange, cv2.COLOR_BGR2RGB)
rotated_monalisa_nearest = cv2.cvtColor(rotated_monalisa_nearest, cv2.COLOR_BGR2RGB)
rotated_monalisa_bilinear = cv2.cvtColor(rotated_monalisa_bilinear, cv2.COLOR_BGR2RGB)
rotated_monalisa_bicubic = cv2.cvtColor(rotated_monalisa_bicubic, cv2.COLOR_BGR2RGB)
rotated_monalisa_lagrange = cv2.cvtColor(rotated_monalisa_lagrange, cv2.COLOR_BGR2RGB)

def plot_comparacao(image_orig, image_scaled, image_rotated, interp_name, prefix, out_dir):
    h_orig, w_orig = image_orig.shape[:2]
    h_scaled, w_scaled = image_scaled.shape[:2]
    h_rot, w_rot = image_rotated.shape[:2]

    background = np.ones_like(image_scaled) * 255
    y_offset = (h_scaled - h_orig) // 2
    x_offset = (w_scaled - w_orig) // 2
    background[y_offset:y_offset+h_orig, x_offset:x_offset+w_orig] = image_orig

    background_rot = np.ones_like(image_scaled) * 255
    y_offset_rot = (h_scaled - h_rot) // 2
    x_offset_rot = (w_scaled - w_rot) // 2
    background_rot[y_offset_rot:y_offset_rot+h_rot, x_offset_rot:x_offset_rot+w_rot] = image_rotated

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(background)
    axs[0].set_title(f"Imagem original")
    axs[0].set_xticks(np.arange(0, w_scaled, 200))
    axs[0].set_yticks(np.arange(0, h_scaled, 200))
    axs[0].grid(True, color='red', linestyle='--', linewidth=0.5)

    axs[1].imshow(image_scaled)
    axs[1].set_title(f"Scaled 2.5({interp_name})")
    axs[1].set_xticks(np.arange(0, w_scaled, 200))
    axs[1].set_yticks(np.arange(0, h_scaled, 200))
    axs[1].grid(True, color='red', linestyle='--', linewidth=0.5)

    axs[2].imshow(background_rot)
    axs[2].set_title(f"Rotated 22.5 ({interp_name})")
    axs[2].set_xticks(np.arange(0, w_scaled, 200))
    axs[2].set_yticks(np.arange(0, h_scaled, 200))
    axs[2].grid(True, color='red', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"comparacao_{prefix}_{interp_name}.png"), dpi=150)
    plt.close(fig)

for interp, interp_name in [
    (scaled_butterfly_nearest, "nearest"),
    (scaled_butterfly_bilinear, "bilinear"),
    (scaled_butterfly_bicubic, "bicubic"),
    (scaled_butterfly_lagrange, "lagrange"),
]:
    if interp_name == "nearest":
        rotated = rotated_butterfly_nearest
    elif interp_name == "bilinear":
        rotated = rotated_butterfly_bilinear
    elif interp_name == "bicubic":
        rotated = rotated_butterfly_bicubic
    elif interp_name == "lagrange":
        rotated = rotated_butterfly_lagrange
    plot_comparacao(image2, interp, rotated, interp_name, "butterfly", out_butterfly)

for interp, interp_name in [
    (scaled_monalisa_nearest, "nearest"),
    (scaled_monalisa_bilinear, "bilinear"),
    (scaled_monalisa_bicubic, "bicubic"),
    (scaled_monalisa_lagrange, "lagrange"),
]:
    if interp_name == "nearest":
        rotated = rotated_monalisa_nearest
    elif interp_name == "bilinear":
        rotated = rotated_monalisa_bilinear
    elif interp_name == "bicubic":
        rotated = rotated_monalisa_bicubic
    elif interp_name == "lagrange":
        rotated = rotated_monalisa_lagrange
    plot_comparacao(image, interp, rotated, interp_name, "monalisa", out_monalisa)