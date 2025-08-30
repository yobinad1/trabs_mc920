import cv2
import numpy as np
from skimage.measure import label, regionprops, regionprops_table
import matplotlib.pyplot as plt

path_in = "/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 4/imgs"
path_out = "/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 4/pt1"

colored_img = cv2.imread(f"{path_in}/objetos3.png")

# 1.1

gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)

gray_img[gray_img < 255] = 0

# 1.2

_, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours_img = np.ones_like(colored_img) * 255

cv2.drawContours(contours_img, contours, -1, (0, 0, 255), 1)

# 1.3

label_img = label(gray_img, background=255)

regions = regionprops(label_img)

n = 0
area_list = []
props_df = regionprops_table(label_img, properties=('area',
                                                 'perimeter',
                                                 'eccentricity',
                                                 'solidity'))

print("número de regiões:", len(regions))

for region in regions:
    print(f"região {n}: área: {region.area} perímetro: {region.perimeter} excentricidade: {region.eccentricity} solidez: {region.solidity}")
    area_list.append(region.area)
    n += 1

display_img = colored_img.copy()

p = 0
for region in regions:
    cY, cX = int(region.centroid[0]), int(region.centroid[1])
    
    text = str(p)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    org = (cX - text_width // 2, cY + text_height // 2)
    
    cv2.putText(display_img, text, org, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    p +=1


# 1.4 (mais fácil de printar com o matplotlib)

small_object = [region for region in regions if region.area < 1500]
medium_object = [region for region in regions if 1500 <= region.area < 3000]
large_object = [region for region in regions if region.area >= 3000]

print("número de regiões pequenas:", len(small_object))
print("número de regiões médias:", len(medium_object))
print("número de regiões grandes:", len(large_object))

plt.hist(area_list, bins=3, color='blue', rwidth=0.9)
plt.xlabel("Área")
plt.ylabel("Número de objetos")
plt.title("Histograma das Áreas")
plt.show()

cv2.imwrite(f"{path_out}/1.1_colored.png", colored_img)
cv2.imwrite(f"{path_out}/1.1_gray.png", gray_img)
cv2.imwrite(f"{path_out}/1.1_binary.png", binary_img)
cv2.imwrite(f"{path_out}/1.1_contours.png", contours_img)
cv2.imwrite(f"{path_out}/1.1_display.png", display_img)