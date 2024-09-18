# Import necessary libraries and packages
import numpy as np
import os
import cv2

class CamShift:
    def __init__(self, video_path):
        self.video_path= video_path

    def detect(self):
        '''Perform CamShift tracking on a video'''

        # Read the video
        cap= cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print('Error: Could not open video.')
            return
        
        # Take the first frame of the video
        ret, frame= cap.read()
        if not ret:
            print('Error: Could not read the first frame.')
            return
        
        # Set up initial location of window
        x, y, w, h= 300, 200, 100, 50 
        track_window= (x, y, w, h)

        # Setup the Region of Interest (ROI) for tracking
        roi=  frame[y:y+h, x:x+w]
        hsv_roi= cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist= cv2.calcHist([hsv_roi], [0], None, [180], [0,180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Setup the termination criteria
        term_crit= (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        
        # Define the output directory for saving frames
        output_dir= 'output/camshift'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        frame_count= 0
        saved_frame_count= 0

        while True:
            ret, frame= cap.read()
            if not ret:
                break

            hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst= cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

            # Apply CamShift to get the new location
            ret, track_window= cv2.CamShift(dst, track_window, term_crit)

            # Draw the tracking result
            pts= cv2.boxPoints(ret)
            pts= np.intp(pts)
            img2= cv2.polylines(frame, [pts], True, 225, 2)

            # Display the result
            cv2.imshow('CamShift Tracking', img2)

            # Save the frame at specified intervals
            if frame_count % 500 == 0:
                output_file= os.path.join(output_dir, f'frame_{saved_frame_count:04d}.png')
                cv2.imwrite(output_file, img2)
                saved_frame_count += 1
                
            frame_count += 1
            
            # Press 'q' to exit the video display
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Testing the CamShift class
if __name__ == '__main__':
    print('Starting CamShift tracking...')

    # Path to the video file
    video_path= 'data/videoplayback.mp4'

    # Create an object of CamShift
    camshift= CamShift(video_path)
    camshift.detect()

    print('CamShift tracking completed.')