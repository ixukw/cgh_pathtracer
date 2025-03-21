import numpy as np
import OpenEXR
import Imath
import matplotlib.pyplot as plt
import cv2
from openexr_numpy import imread, imwrite


# Function to load the phase data from the .exr file
def load_phase_from_exr(filename):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(filename)
    
    # Get the data window to extract dimensions
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Read the phase channel (assuming phase is stored in the 'R' channel)
    #phase_channel = exr_file.channel('R')
    phase_channel = imread(filename, "R")
    
    # Convert the channel into a numpy array
    #phase = np.frombuffer(phase_channel, dtype=np.float32)
    print(phase_channel.shape)
    
    return phase_channel, width, height

# Function to generate the hologram and compute the Fourier transform
def generate_image_plane(phase, width, height, lambda_val, z, p):
    # Step 1: Create the complex hologram E(u, v) = exp(i * phase(u, v))
    hologram = np.exp(1j * phase)

    # Step 2: Compute the 2D Fourier transform (F(t) = FFT2(E))
    ft_hologram = np.fft.fftshift(np.fft.fft2(hologram))

    # Step 3: Compute the intensity (magnitude squared) of the Fourier transform
    intensity = np.abs(ft_hologram) ** 2

    # Step 4: Normalize the intensity and convert to a range of 0-255 for display
    intensity = intensity / np.max(intensity) * 255
    intensity = np.uint8(intensity)

    # Step 5: Save or display the hologram intensity as a grayscale image
    cv2.imwrite('hologram_image.png', intensity)  # Save as image
    plt.imshow(intensity, cmap='gray')
    plt.colorbar()
    plt.show()

# Main function to run the process
def main():
    # File path to the phase .exr file
    phase_file = 'vol_cbox1.exr'

    # Load the phase data from the .exr file
    phase, width, height = load_phase_from_exr(phase_file)
    print(width, height)
    # Parameters for the holography simulation
    lambda_val = 0.633e-6  # Wavelength in meters
    z = 0.1  # Propagation distance in meters
    p = 10e-6  # Pixel pitch in meters

    # Generate the image plane at the hologram
    generate_image_plane(phase, width, height, lambda_val, z, p)

if __name__ == '__main__':
    main()
