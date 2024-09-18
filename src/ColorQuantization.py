# Import necessary libraries and packages
import numpy as np
import os
import cv2

class ColorQuantization:
    def __init__(self, image_path):
        self.image_path= image_path

    def quantize(self, k= 8):
        '''Perform Color Quantization on an image using K-Means Clustering'''

        # Read the image
        image= cv2.imread(self.image_path)
        if image is None:
            print('Error: Image not found.')
            return None
        
        # Convert the image to a 2D array of pixels
        Z= image.reshape((-1,3))
        Z= np.float32(Z)

        # Define criteria and apply K-Means
        criteria= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center= cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert the center values back to 8-bit values
        center= np.uint8(center)

        # Map the labels to the center values
        res= center[label.flatten()]
        quantized_image= res.reshape((image.shape))

        # Display the original and quantized image
        cv2.imshow('Original Image', image)
        cv2.imshow('Quantized Image', quantized_image)

        # Save the quantized image to the output directory
        output_dir= 'output/color_quantization'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file= os.path.join(output_dir, 'quantized_image.png')
        cv2.imwrite(output_file, quantized_image)
        print(f'Quantized image saved at {output_file}')

        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Testing the ColorQuantization class
if __name__ == '__main__':
    print('Starting color quantization...')

    # Path to the image
    image_path= 'data/download.jpeg'

    # Create an object of ColorQuantization
    color_quantization= ColorQuantization(image_path)
    color_quantization.quantize(k= 8)

    print('Color quantization completed.')