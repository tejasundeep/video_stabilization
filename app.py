import cv2
import numpy as np

# Constants
FOLDER_TEMPLATE = 'data/folder/{0}'
VIDEO_INPUT_PATH = FOLDER_TEMPLATE.format('video.wmv')
VIDEO_OUTPUT_PATH = FOLDER_TEMPLATE.format('output.mp4')
RESET_FREQUENCY = 100

def get_video_capture(path):
    """Returns a cv2.VideoCapture object for the given path."""
    return cv2.VideoCapture(path)

def get_initial_frame(video_capture):
    """Returns the initial frame from the video capture object."""
    ret, frame = video_capture.read()
    if not ret or frame is None:
        raise ValueError("Could not read the initial frame.")
    return frame[:, :, 0]

def get_frame_transforms(video_capture, initial_frame):
    """Calculates and returns frame-to-frame transformation matrices."""
    frames = [initial_frame]
    transforms = [np.identity(3)]
    while True:
        ret, current_frame = video_capture.read()
        if not ret or current_frame is None:
            break
        current = current_frame[:, :, 0]
        prev_corner = cv2.goodFeaturesToTrack(frames[-1], 200, 0.0001, 10)
        cur_corner, status, _ = cv2.calcOpticalFlowPyrLK(frames[-1], current, prev_corner, None)
        prev_corner, cur_corner = [corners[status.ravel().astype(bool)] for corners in [prev_corner, cur_corner]]
        transform = cv2.estimateRigidTransform(prev_corner, cur_corner, True)
        transform = np.append(transform, [[0, 0, 1]], axis=0) if transform is not None else transforms[-1]
        transforms.append(transform)
        frames.append(current)
    return frames, transforms

def stabilize_frames(frames, transforms):
    """Stabilizes frames using the calculated transforms."""
    height, width = frames[0].shape
    stabilized_frames = []
    last_transform = np.identity(3)
    for index, (frame, transform) in enumerate(zip(frames, transforms)):
        transform = np.dot(transform, last_transform)
        if index % RESET_FREQUENCY == 0:
            transform = np.identity(3)
        last_transform = transform
        inverse_transform = cv2.invertAffineTransform(transform[:2])
        stabilized_frames.append(cv2.warpAffine(frame, inverse_transform, (width, height)))
    return stabilized_frames

def write_output_video(frames, stabilized_frames, path):
    """Writes the original and stabilized frames side by side to the output video."""
    height, width = frames[0].shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('H','2','6','4'), 20.0, (width*2, height), False)
    for frame, stabilized in zip(frames, stabilized_frames):
        writer.write(np.concatenate([stabilized, frame], axis=1))
    writer.release()

def main():
    """Main function to orchestrate the video stabilization process."""
    video_capture = get_video_capture(VIDEO_INPUT_PATH)
    initial_frame = get_initial_frame(video_capture)
    frames, transforms = get_frame_transforms(video_capture, initial_frame)
    video_capture.release()
    
    stabilized_frames = stabilize_frames(frames, transforms)
    write_output_video(frames, stabilized_frames, VIDEO_OUTPUT_PATH)

if __name__ == "__main__":
    main()
