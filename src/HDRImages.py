# Import necessary libraries and packages
import numpy as np
import os
import cv2

class HDRImaging:
    def __init__(self, hdr_images_dir):
        self.hdr_images_dir= hdr_images_dir

    def create_hdr_image(self):
        '''Create HDR Image from a set of images'''

        # Collect all images from the directory and their exposure times
        image_files= []
        exposure_times= []

        with open(os.path.join(self.hdr_images_dir, 'exposure_times.txt'), 'r') as file:
            for line in file:
                img_name, exposure_time= line.strip().split()
                image_files.append(os.path.join(self.hdr_images_dir, img_name))
                exposure_times.append(float(exposure_time))

        images= [cv2.imread(img_file) for img_file in image_files]

        if not images:
            print('Error: No images found in the specified directory.')
            return
        
        exposure_times= np.array(exposure_times, dtype= np.float32)

        # Use AlignMTB algorithm to align images
        alignMTB= cv2.createAlignMTB()
        alignMTB.process(images, images)

        # Merge to HDR using exposure times
        merge_debevec= cv2.createMergeDebevec()
        hdr= merge_debevec.process(images, times= exposure_times.copy())

        # Tonemap the HDR image to convert it to an 8-bit image
        tonemap= cv2.createTonemap(gamma= 2.2)
        ldr= tonemap.process(hdr)

        # Normalize to the range 0-255 for display
        ldr_8bit= np.clip(ldr * 255, 0, 255).astype('uint8')

        # Display the result
        cv2.imshow('HDR Image', ldr_8bit)

        # Save the HDR image to the output directory
        output_dir= 'output/hdr_imaging'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file= os.path.join(output_dir, 'hdr_image.png')
        cv2.imwrite(output_file, ldr_8bit)
        print(f'HDR image saved at {output_file}')

        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Test the HDRImaging class
if __name__ == '__main__':
    print('Staring HDR imaging...')

    # Path to the HDR image directory
    hdr_images_dir= 'data/HDRImagesInput'

    # Create an object of HDRImaging
    hdr_imaging= HDRImaging(hdr_images_dir)
    hdr_imaging.create_hdr_image()

    print('HDR imaging completed.')