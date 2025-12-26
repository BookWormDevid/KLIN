"""
1) YOLO = 11n
2) OpenPose
3) LSTM? Use LM studio perhaps?
4) ActionFormer
"""

import cv2
from ultralytics import YOLO


def find_available_cameras(max_tests=5):
    """Find available camera indices"""
    available_cameras = []
    for i in range(max_tests):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras


def initialize_video_source(source=None):
    """
    Initialize video source - tries camera first, then falls back to video file
    """
    # If source is provided, use it
    if source is not None:
        if isinstance(source, int):
            # Camera index
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                return cap, f"Camera {source}"
        else:
            # Video file path
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                return cap, f"Video file: {source}"

    # Auto-detect camera
    print("Searching for available cameras...")
    available_cams = find_available_cameras()

    if available_cams:
        cam_index = available_cams[0]
        cap = cv2.VideoCapture(cam_index)
        return cap, f"Auto-detected Camera {cam_index}"

    # Fallback to sample video or prompt user
    print("No cameras found. Please provide a video file path.")
    video_path = input("Enter video file path: ")
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap, f"Video file: {video_path}"
    else:
        raise Exception("No video source available")


def real_time_pose_detection(source=None, model_name='yolo11n-pose.pt'):
    # Load model
    model = YOLO(model_name)

    # Initialize video source
    cap, source_info = initialize_video_source(source)
    print(f"Using: {source_info}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break

            # Resize for better performance (optional)
            frame = cv2.resize(frame, (640, 480))

            # Run inference
            results = model(frame, conf=0.5, verbose=False)

            # Annotate frame with results
            annotated_frame = results[0].plot()

            # Add source info
            cv2.putText(annotated_frame, source_info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display
            cv2.imshow('YOLO Pose Detection', annotated_frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# Usage examples:
# Auto-detect camera
real_time_pose_detection()

# Use specific camera
# real_time_pose_detection(source=0)

# Use video file
# real_time_pose_detection(source='your_video.mp4')
