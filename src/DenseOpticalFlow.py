# Import necessary libraries and packages
import numpy as np
import os
import cv2

class DenseOpticalFlow:
    def __init__(self, video_path):
        self.video_path= video_path

    def dense_optical_flow(self, frame_interval= 500):
        '''Perform Dense Optical Flow on a video and save frames at intervals'''

        # Read the video
        cap= cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print('Error: Could not open video.')
            return
        
        # Read the first frame
        ret, frame1= cap.read()
        if not ret:
            print('Error: Could not read the first frame.')
            return
        
        # Convert the frame to grayscale
        prvs= cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # Define the output directory for saving frames
        output_dir= 'output/dense_optical_flow'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        frame_count= 0
        saved_frame_count= 0

        while True:
            ret, frame2= cap.read()
            if not ret:
                break

            # Conver the frame to grayscale
            next= cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Calculate Dense Optical Flow using the Farneback method
            flow= cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Convert flow to RGB image
            hsv= cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
            mag, ang= cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0]= ang * 180 / np.pi / 2
            hsv[..., 1]= 255
            hsv[..., 2]= cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            dense_flow_img= cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Save the frame at specified intervals
            if frame_count % frame_interval == 0:

                # Display the result
                cv2.imshow('Dense Optical Flow', dense_flow_img)

                # Save the frame to the output directory
                output_file= os.path.join(output_dir, f'frame_{saved_frame_count:04d}.png')
                cv2.imwrite(output_file, dense_flow_img)
                saved_frame_count += 1

                # Press 'q' to exit the video display
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

            frame_count += 1
            prvs= next

        cap.release()
        cv2.destroyAllWindows()

# Testing the DenseOpticalFlow class
if __name__ == '__main__':
    print('Starting dense optical flow...')

    # Path to the video file
    video_path= 'data/videoplayback.mp4'

    # Create an object of DenseOpticalFlow
    dense_optical_flow= DenseOpticalFlow(video_path)
    dense_optical_flow.dense_optical_flow(frame_interval= 5000)

    print('Dense optical flow completed.')