import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def dist3D(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def angle_xz(start, end):
    dx = end[0] - start[0]
    dz = end[2] - start[2]
    angle_rad = np.arctan2(dz, dx)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def swing_path_direction_side(start, end):
    dz = end[2] - start[2]
    return dz

cap = cv2.VideoCapture("vivaanswing.mp4")

recording = False
positions = []
start_time = 0
max_speed = 0

print("Raise your right hand to start recording the swing.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror so right side is right side of image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    h, w, _ = frame.shape

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Draw all landmarks and connections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw landmark indices next to each joint
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.putText(frame, str(i), (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Right wrist (16) and right elbow (14)
        rw = landmarks[16]
        re = landmarks[14]
        rw_coords = (rw.x, rw.y, rw.z)

        rw_px = (int(rw.x * w), int(rw.y * h))

        # Detect hand raised (wrist above elbow in image coordinates)
        if not recording and rw.y < re.y - 0.1:
            recording = True
            positions = []
            start_time = time.time()
            max_speed = 0
            print("[START] Swing recording started!")

        if recording:
            positions.append(rw_coords)

            if len(positions) > 1:
                p1 = np.array(positions[-2])
                p2 = np.array(positions[-1])
                dt = 1 / 30
                speed = dist3D(p1, p2) / dt
                if speed > max_speed:
                    max_speed = speed

            if len(positions) > 10:
                recent_speeds = [dist3D(np.array(positions[i]), np.array(positions[i+1])) / dt for i in range(-10, -1)]
                avg_speed = sum(recent_speeds) / len(recent_speeds)
                if avg_speed < 0.001 or (time.time() - start_time) > 5:
                    recording = False

                    start_pos = positions[5]
                    end_pos = positions[-5]

                    dz = swing_path_direction_side(start_pos, end_pos)
                    angle = angle_xz(start_pos, end_pos)

                    print("\n=== Swing Analysis (Side View) ===")
                    print(f"Peak clubhead speed (normalized units/s): {max_speed:.2f}")
                    print(f"Final swing path angle (degrees): {angle:.2f}")

                    if dz < -0.02:
                        print("Prediction: Slice (outside-in path)")
                    elif -0.02 <= dz < -0.005:
                        print("Prediction: Fade (slight outside-in)")
                    elif -0.005 <= dz <= 0.005:
                        print("Prediction: Straight shot")
                    elif 0.005 < dz <= 0.02:
                        print("Prediction: Draw (slight inside-out)")
                    else:
                        print("Prediction: Hook (strong inside-out path)")
                    print("===============================\n")

        cv2.circle(frame, rw_px, 10, (0, 255, 0) if recording else (0, 0, 255), -1)

    cv2.putText(frame, "Raise right hand to start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.imshow("Golf Swing Tracker - Side View", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
