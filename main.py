import cv2
import mediapipe as mp
import numpy as np
import time

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 = webcam

recording = False
positions = []
start_time = 0
fps = 30  # You can measure this dynamically

def estimate_speed(positions, fps, px_per_meter=300):
    speeds = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        dist_px = np.sqrt(dx ** 2 + dy ** 2)
        speed_mps = (dist_px / px_per_meter) * fps
        speeds.append(speed_mps)
    return speeds

def swing_path_angle(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = np.arctan2(dy, dx) * 180 / np.pi
    return angle

print("Raise your right hand above your head to begin swing recording.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = result.pose_landmarks.landmark
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        h, w, _ = frame.shape
        wrist_px = (int(right_wrist.x * w), int(right_wrist.y * h))
        shoulder_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))

        # Trigger recording when right hand is raised above shoulder
        if not recording and right_wrist.y < right_shoulder.y:
            print("Swing recording started!")
            recording = True
            positions = []
            start_time = time.time()

        # Record wrist positions while swinging
        if recording:
            positions.append(wrist_px)
            cv2.circle(frame, wrist_px, 6, (0, 255, 0), -1)

            # Stop recording when swing slows down (velocity drop)
            if len(positions) >= 10:
                dx = positions[-1][0] - positions[-2][0]
                dy = positions[-1][1] - positions[-2][1]
                velocity = np.sqrt(dx**2 + dy**2)
                if velocity < 2.5 and (time.time() - start_time > 0.75):  # tweak thresholds
                    recording = False
                    print("Swing recording stopped.")

                    # Analyze swing
                    speeds = estimate_speed(positions, fps)
                    max_speed = max(speeds) if speeds else 0

                    start = positions[5]
                    end = positions[-5]
                    angle = swing_path_angle(start, end)

                    print("\n=== Swing Analysis ===")
                    print(f"Peak clubhead speed: {max_speed:.2f} m/s")
                    print(f"Swing path angle: {angle:.2f}°")

                    if angle < -10:
                        print("Prediction: Slice (outside-in path)")
                    elif angle > 10:
                        print("Prediction: Draw (inside-out path)")
                    else:
                        print("Prediction: Straight shot")
                    print("======================\n")

    # Show frame
    cv2.imshow("Real-Time Golf Swing Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
