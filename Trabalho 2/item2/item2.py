import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# loading watch image
path = "/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 2/imgs/baboon_monocromatica.png"
path_out = '/home/daniel/Documents/Unicamp/IC/MC920/Trabalho 2/item2'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# applying fft
fft_image = np.fft.fft2(image)

# shifting the zero frequency component to the center
fft_image_shifted = np.fft.fftshift(fft_image)

# printar o espectro de magnitude
magnitude_spectrum = np.abs(fft_image_shifted)
magnitude_spectrum = np.log(magnitude_spectrum + 1)
magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum) * 255
magnitude_spectrum = np.uint8(magnitude_spectrum)

# take the shape of the image
r, c = image.shape
# take the center of the image
cr, cc = r // 2, c // 2

# creating all filters
def low_pass_filter(shape, rad):
    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, (cc, cr), rad, 1, thickness=-1)
    return mask

def high_pass_filter(shape, rad):
    mask = np.ones(shape, np.uint8)
    cv2.circle(mask, (cc, cr), rad, 0, thickness=-1)
    return mask

def band_pass_filter(shape, rad_in, rad_out):
    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, (cc, cr), rad_out, 1, thickness=-1)
    cv2.circle(mask, (cc, cr), rad_in, 0, thickness=-1)
    return mask

def band_stop_filter(shape, rad_in, rad_out):
    mask = np.ones(shape, np.uint8)
    cv2.circle(mask, (cc, cr), rad_out, 0, thickness=-1)
    cv2.circle(mask, (cc, cr), rad_in, 1, thickness=-1)
    return mask

# setting radius values
rad_in = 150
rad_out = 230

mask_pb = low_pass_filter(image.shape, rad_in)
mask_pa = high_pass_filter(image.shape, rad_in)
mask_pf = band_pass_filter(image.shape, rad_in, rad_out)
mask_rf = band_stop_filter(image.shape, rad_in, rad_out)

# function to apply filters
def apply_filter(fft_image_shifted, mask):
    image_filtered = fft_image_shifted * mask
    inverse_fft = np.fft.ifftshift(image_filtered)
    filtered_inversed_image = np.fft.ifft2(inverse_fft)
    filtered_inversed_image = np.abs(filtered_inversed_image)
    return filtered_inversed_image

# applying filters to the image
img_pb = apply_filter(fft_image_shifted, mask_pb)
img_pa = apply_filter(fft_image_shifted, mask_pa)
img_pf = apply_filter(fft_image_shifted, mask_pf)
img_rf = apply_filter(fft_image_shifted, mask_rf)

# saving images
cv2.imwrite(os.path.join(path_out, 'imagem_pb.png'), img_pb)
cv2.imwrite(os.path.join(path_out, 'imagem_pa.png'), img_pa)
cv2.imwrite(os.path.join(path_out, 'imagem_pf.png'), img_pf)
cv2.imwrite(os.path.join(path_out, 'imagem_rf.png'), img_rf)
cv2.imwrite(os.path.join(path_out, 'spectrum.png'), magnitude_spectrum)

# saving filtered spectrum images
filtered_spectrum_pb = np.abs(fft_image_shifted * mask_pb)
filtered_spectrum_pb = np.log(filtered_spectrum_pb + 1)
filtered_spectrum_pb = filtered_spectrum_pb / np.max(filtered_spectrum_pb) * 255
filtered_spectrum_pb = np.uint8(filtered_spectrum_pb)
cv2.imwrite(os.path.join(path_out, 'spectrum_pb.png'), filtered_spectrum_pb)

filtered_spectrum_pa = np.abs(fft_image_shifted * mask_pa)
filtered_spectrum_pa = np.log(filtered_spectrum_pa + 1)
filtered_spectrum_pa = filtered_spectrum_pa / np.max(filtered_spectrum_pa) * 255
filtered_spectrum_pa = np.uint8(filtered_spectrum_pa)
cv2.imwrite(os.path.join(path_out, 'spectrum_pa.png'), filtered_spectrum_pa)

filtered_spectrum_pf = np.abs(fft_image_shifted * mask_pf)
filtered_spectrum_pf = np.log(filtered_spectrum_pf + 1)
filtered_spectrum_pf = filtered_spectrum_pf / np.max(filtered_spectrum_pf) * 255
filtered_spectrum_pf = np.uint8(filtered_spectrum_pf)
cv2.imwrite(os.path.join(path_out, 'spectrum_pf.png'), filtered_spectrum_pf)

filtered_spectrum_rf = np.abs(fft_image_shifted * mask_rf)
filtered_spectrum_rf = np.log(filtered_spectrum_rf + 1)
filtered_spectrum_rf = filtered_spectrum_rf / np.max(filtered_spectrum_rf) * 255
filtered_spectrum_rf = np.uint8(filtered_spectrum_rf)
cv2.imwrite(os.path.join(path_out, 'spectrum_rf.png'), filtered_spectrum_rf)

# compressing the image using FFT
def compress_fft(fft_image, threshold=0.005):
    magnitude = np.abs(fft_image)
    compressed = np.where(magnitude > threshold * np.max(magnitude), fft_image, 0)
    return compressed

# compressing the image
compressed_fft = compress_fft(fft_image_shifted)
compressed_image = np.fft.ifft2(np.fft.ifftshift(compressed_fft))
compressed_image = np.abs(compressed_image)
cv2.imwrite(os.path.join(path_out, 'imagem_compressao.png'), compressed_image)

# saving the compressed spectrum
def plot_histogram(image, filename):
    plt.figure()
    plt.hist(image.ravel(), bins=256, color='darkred', alpha=0.7, histtype='stepfilled', edgecolor='none')
    plt.xlim([0, 256])
    plt.savefig(os.path.join(path_out, filename))
    plt.close()

plot_histogram(image, 'histograma_original.png')
plot_histogram(compressed_image, 'histograma_comprimida.png')
