# Import necessary libraries and packages
import numpy as np
import os
import cv2

class EpipolarGeometry():
    def __init__(self, left_image_path, right_image_path):
        self.left_image_path= left_image_path
        self.right_image_path= right_image_path

    def detect(self):
        '''Compute Epipolar Geometry using stereo images'''

        # Read the stereo images
        img_left= cv2.imread(self.left_image_path)
        img_right= cv2.imread(self.right_image_path)

        if img_left is None or img_right is None:
            print('Error: One or both images not found.')
            return
        
        # Convert images to grayscale for SIFT
        img_left_gray= cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right_gray= cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # Initialize the SIFT detector
        sift= cv2.SIFT_create()

        # Find keypoints and descriptors with SIFT
        keypoints_left, descriptors_left= sift.detectAndCompute(img_left_gray, None)
        keypoints_right, descriptors_right= sift.detectAndCompute(img_right_gray, None)

        # Feature matching
        bf= cv2.BFMatcher(cv2.NORM_L2, crossCheck= True)
        matches= bf.match(descriptors_left, descriptors_right)
        matches= sorted(matches, key= lambda x: x.distance)

        # Draw the first 10 matches
        img_matches= cv2.drawMatches(
            img_left, 
            keypoints_left,
            img_right, 
            keypoints_right,
            matches[:10],
            None,
            flags= cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Display the matches
        cv2.imshow('Epipolar Geometry - Matches', img_matches)

        # Save the matches image to the output directory
        output_dir= 'output/epipolar_geometry'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file= os.path.join(output_dir, 'epipolar_matches.png')
        cv2.imwrite(output_file, img_matches)
        print(f'Matches image saved at {output_file}')

        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Testing the EpipolarGeometry class
if __name__ == '__main__':
    print('Starting epipolar geometry computation...')

    # Paths to the stereo images
    left_image_path= 'data/view0.png'
    right_image_path= 'data/view2.png'

    # Create an object of EpipolarGeometry
    epipolar_geometry= EpipolarGeometry(left_image_path, right_image_path)
    epipolar_geometry.detect()

    print('Epipolar geometry computation completed.')