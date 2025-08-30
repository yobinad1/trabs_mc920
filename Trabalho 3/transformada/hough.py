import cv2
import numpy as np
import os
import pytesseract

def detect_skew_hough(image):
    # applying canny edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    threshold = 200
    while threshold > 50:  # minimun threshold
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
        if lines is not None:
            break
        threshold -= 50
        
    if lines is None:
        print("Nenhuma linha detectada. Retornando ângulo 0.")
        return 0.0  # lines not detected, return 0 angle
    
    # finding the angles of the detected lines
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.rad2deg(theta) - 90
        angles.append(angle)

    # calculate the median angle
    skew_angle = np.median(angles)
    
    return skew_angle   
    
def correct_skew(image, angle):
    h, w = image.shape
    center = (w // 2, h // 2)
    
    # rotating the image to correct the skew
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    corrected_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    return corrected_image
    
# paths of input and output images
path_in = "/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 3/imgs"
path_out = '/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 3/transformada'

# processing all images in the input directory
for filename in os.listdir(path_in):
    if filename.lower().endswith(('.png')):  # Check for image files
        # input path
        input_path = os.path.join(path_in, filename)

        # reading image
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # skew angle detection
        skew_angle = detect_skew_hough(image)

        # correcting the skew
        corrected_image = correct_skew(image, skew_angle)

        # extract text using OCR
        original_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
        corrected_text = pytesseract.image_to_string(corrected_image, lang='eng', config='--psm 6')

        # create a directory for the current image outputs
        image_name = os.path.splitext(filename)[0]
        image_output_dir = os.path.join(path_out, image_name)
        os.makedirs(image_output_dir, exist_ok=True)

        # save the corrected image
        corrected_image_path = os.path.join(image_output_dir, f"corrected_{filename}")
        cv2.imwrite(corrected_image_path, corrected_image)

        # save the texts to .txt files
        original_text_path = os.path.join(image_output_dir, "original_text.txt")
        corrected_text_path = os.path.join(image_output_dir, "corrected_text.txt")

        with open(original_text_path, 'w', encoding='utf-8') as file:
            file.write(original_text)

        with open(corrected_text_path, 'w', encoding='utf-8') as file:
            file.write(corrected_text)

        # save the skew angle to a .txt file
        angle_path = os.path.join(image_output_dir, "skew_angle.txt")
        with open(angle_path, 'w', encoding='utf-8') as file:
            file.write(f"Skew angle: {skew_angle:.2f} degrees")

        # print the results for each image in terminal
        print(f"Processed {filename}")
        print(f"Ângulo de inclinação detectado: {skew_angle:.2f} graus")