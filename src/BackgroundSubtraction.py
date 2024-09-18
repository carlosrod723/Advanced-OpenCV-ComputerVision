# Import necessary libraries and packages
import os
import cv2

class BackgroundSubtraction:
    def __init__(self, video_path):
        self.video_path= video_path

    def background_subtract(self, frame_interval= 500):
        '''Perform Background Subtraction on a video and save frames at intervals'''

        # Read the video
        cap= cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print('Error: Could  not open video.')
            return
        
        # Create the background subtractor
        fgbg= cv2.createBackgroundSubtractorMOG2()
        
        # Define the output directory for saving frames
        output_dir= 'output/background_subtraction'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        frame_count= 0
        saved_frame_count= 0
        
        while True:
            ret, frame= cap.read()
            if not ret:
                break
            
            # Apply the background subtractor
            fgmask= fgbg.apply(frame)
            
            # Save the frame at specified intervals
            if frame_count % frame_interval == 0:

                # Display the result
                cv2.imshow('Frame', fgmask)

                # Save the frame to the output directory
                output_file= os.path.join(output_dir, f'frame_{saved_frame_count:04d}.png')
                cv2.imwrite(output_file, fgmask)
                saved_frame_count += 1

                # Press 'q' to exit the video display
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

# Testing the BackgroundSubtraction class
if __name__ == '__main__':
    print('Starting background subtraction...')

    # Path to the video file
    video_path= 'data/videoplayback.mp4'

    # Create an object of BackgroundSubtraction
    background_subtractor= BackgroundSubtraction(video_path)
    background_subtractor.background_subtract(frame_interval= 5000)

    print('Background Subtraction completed.')