# Import necessary libraries and packages
import numpy as np
import os
import cv2

class DepthMapStereo:
    def __init__(self, left_image_path, right_image_path):
        self.left_image_path= left_image_path
        self.right_image_path= right_image_path

    def compute_depth_map(self):
        '''Compute the depth map using stereo images'''

        # Read the stereo images
        img_left= cv2.imread(self.left_image_path, cv2.IMREAD_GRAYSCALE)
        img_right= cv2.imread(self.right_image_path, cv2.IMREAD_GRAYSCALE)

        if img_left is None or img_right is None:
            print('Error: One or both images not found.')
            return 
        
        # Stereo Block Matching (BM) algorithm
        stereo= cv2.StereoBM_create(numDisparities= 16, blockSize= 15)
        disparity= stereo.compute(img_left, img_right)

        # Normalize the disparity for visualization
        disp= cv2.normalize(disparity, disparity, alpha= 0, beta= 255, norm_type= cv2.NORM_MINMAX)
        disp= np.uint8(disp)

        # Display the depth map
        cv2.imshow('Depth Map', disp)

        # Save the depth map to the output directory
        output_dir= 'output/depth_map_stereo'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file= os.path.join(output_dir, 'depth_map.png')
        cv2.imwrite(output_file, disp)
        print(f'Depth map saved at {output_file}')

        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Testing the DepthMapStereo class
if __name__ == '__main__':
    print('Starting depth map computation...')

    # Paths to the stereo images
    left_image_path= 'data/view0.png'
    right_image_path= 'data/view2.png'

    # Create an object of DepthMapStereo
    depth_map_stereo= DepthMapStereo(left_image_path, right_image_path)
    depth_map_stereo.compute_depth_map()

    print('Depth map comptuation completed.')