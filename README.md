# video_stabilization

Getting Frame-to-Frame Transform Matrices:
The video file 'video.wmv' is opened using cv2.VideoCapture.
The first frame of the video is read and stored in the frames list.
The transforms list is initialized with an identity matrix.
A while loop runs to read each frame from the video and calculate the transformation matrix between the current and the previous frame using optical flow and feature tracking methods (cv2.goodFeaturesToTrack and cv2.calcOpticalFlowPyrLK).
If a valid transformation matrix is found, it is stored; otherwise, the last valid transformation matrix is reused.


Using the Transforms to Stabilize Images:
A new list stabilized_frames is created to store the stabilized frames.
A for loop iterates over each frame and its corresponding transformation matrix.
Inside the loop, the transformation matrices are accumulated to get the transformation from the current frame to the initial frame.
Every reset_frequency frames, the transformation is reset to the identity matrix to prevent drift over time.
The inverse of the accumulated transformation is used to warp the current frame to a stabilized position using cv2.warpAffine.


Writing the Output Video:
A video writer object is created using cv2.VideoWriter to write the output video as an MP4 file with H.264 codec.
A for loop iterates over the original and stabilized frames, concatenating them side by side, and writes them to the output video.
Finally, the writer object is released to save the video to the disk.


The output video will show the original and stabilized videos side by side, allowing you to visually compare the stabilization effect. The stabilized video should have reduced shake compared to the original video, but depending on the video content and the reset_frequency parameter, there might be some artifacts or issues in the stabilization. Adjusting parameters like the feature detection thresholds and reset_frequency might help in getting better results.
