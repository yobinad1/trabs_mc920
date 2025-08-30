import cv2
import numpy as np
import os
import pytesseract

# function to calculate the projection profile of the image
def profile_projection(image):
    return np.sum(image, axis=1)

# function to calculate the objective function for the projection profile
def objective_function(profile):
    return np.sum(np.diff(profile) ** 2)

# function to detect the skew angle using the projection profile method
def detect_skew_angle(image, theta_min=-90, theta_max=90, step=0.1):
    best_angle = 0
    max_value = -np.inf

    for theta in np.arange(theta_min, theta_max + step, step):
        h, w = image.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, theta, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        profile = profile_projection(rotated)
        value = objective_function(profile)

        if value > max_value:
            max_value = value
            best_angle = theta

    return best_angle 

# paths of input and output images
path_in = "/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 3/imgs"
path_out = '/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 3/projeção'

for filename in os.listdir(path_in):
    if filename.lower().endswith(('.png')):  # Check for image files
        # input path
        input_path = os.path.join(path_in, filename)

        # reading image
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # skew angle detection
        skew_angle = detect_skew_angle(image)

        # correcting the skew
        h, w = image.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        corrected_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

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