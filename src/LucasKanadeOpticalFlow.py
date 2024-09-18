# Import necessary libraries and packages
import numpy as np
import os
import cv2

class LucasKanadeOpticalFlow:
    def __init__(self, video_path):
        self.video_path= video_path

    def detect(self, frame_interval= 500):
        '''Perform Lucas Kanade Optical Flow on a video and save frames at an interval'''

        # Read the video
        cap= cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print('Error: Could not open video.')
            return
        
        # Define parameters for ShiTomasi corner detection
        feature_params= dict(
            maxCorners=100,
            qualityLevel= 0.3,
            minDistance= 7, 
            blockSize= 7
        )
        
        # Define parameters for Lucas-Kanade Optical Flow
        lk_params= dict(
            winSize= (15,15),
            maxLevel= 2,
            criteria= (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Take the first frame and find corners in it
        ret, old_frame= cap.read()
        if not ret:
            print('Error: Could not read the first frame.')
            return
        
        old_gray= cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0= cv2.goodFeaturesToTrack(old_gray, mask= None, **feature_params)

        # Create a mask image for drawing purposes
        mask= np.zeros_like(old_frame)

        # Define the output directory for saving frames
        output_dir= 'output/lucas_kanade_optical_flow'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        frame_count= 0
        saved_frame_count= 0

        while True:
            ret, frame= cap.read()
            if not ret:
                break

            frame_gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate the optical flow
            p1, st, err= cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Check if p1 is None
            if p1 is None:
                print('Warning: No optical flow found in this frame.')
                break

            # Select good points
            good_new= p1[st == 1]
            good_old= p0[st == 1]

            # Draw the tracks
            for i, (new,old) in enumerate(zip(good_new, good_old)):
                a, b= new.ravel()
                c, d= old.ravel()
                mask= cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0,255,0), 2)
                frame= cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            img= cv2.add(frame, mask)

            # Save the frame at specified intervals
            if frame_count % frame_interval == 0:

                # Display the result
                cv2.imshow('Lucas Kanade Optical Flow', img)

                # Save the frame to the output directory
                output_file= os.path.join(output_dir, f'frame_{saved_frame_count:04d}.png')
                cv2.imwrite(output_file, img)
                saved_frame_count += 1

                # Press 'q' to exit the video display
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

            frame_count += 1

            # Update the previous frame and previous points
            old_gray= frame_gray.copy()
            p0= good_new.reshape(-1,1,2)

        cap.release()
        cv2.destroyAllWindows()

# Testing the LucasKanadeOpticalFlow class
if __name__ == '__main__':
    print('Starting Lucas Kanade optical flow...')

    # Path to the video file
    video_path= 'data/videoplayback.mp4'

    # Create an object of LucasKanadeOpticalFlow
    lucas_kanade_optical_flow= LucasKanadeOpticalFlow(video_path)
    lucas_kanade_optical_flow.detect(frame_interval= 5000)

    print('Lucas Kanade optical flow completed.')