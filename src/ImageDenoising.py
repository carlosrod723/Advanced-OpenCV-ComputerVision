# Import necessary libraries and packages
import numpy as np
import os
import cv2

class ImageDenoising:
    def __init__(self, image_path):
        self.image_path= image_path

    def denoise(self):
        '''Apply different denoising techniques to the image'''

        # Read the image
        image= cv2.imread(self.image_path)
        if image is None:
            print('Error: Image not found.')
            return
        
        # Apply FastNlMeansDenoisingColored
        denoised_image_colored= cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Convert image to grayscale
        grayscale_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply FastNlMeansDenoising on grayscale image
        denoised_image_gray= cv2.fastNlMeansDenoising(grayscale_image, None, 10, 7, 21)

        # Display the original and denoised image
        cv2.imshow('Original Image', image)
        cv2.imshow('Denoised Image- Colored', denoised_image_colored)
        cv2.imshow('Denoised Image- Grayscale', denoised_image_gray)

        # Save the denoised images to the output directory
        output_dir= 'output/image_denoising'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the images
        output_file_colored= os.path.join(output_dir, 'denoised_colored.png')
        cv2.imwrite(output_file_colored, denoised_image_colored)
        print(f'Denoised colored image saved at {output_file_colored}')

        output_file_gray= os.path.join(output_dir, 'denoised_gray.png')
        cv2.imwrite(output_file_gray, denoised_image_gray)
        print(f'Denoised grayscale imaged saved at {output_file_gray}')

        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Testing the ImageDenoising class
if __name__ == '__main__':
    print('Starting image denoising...')

    # Path to the image
    image_path= 'data/view0.png'

    # Create an object of ImageDenoising
    image_denoising= ImageDenoising(image_path)
    image_denoising.denoise()

    print('Image denoising completed.')