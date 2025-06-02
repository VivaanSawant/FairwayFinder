import cv2
import numpy as np

# Load video (replace with 0 for webcam)
cap = cv2.VideoCapture("golf_swing.mp4")

# Ball tracking variables
ball_positions = []
trajectory_length = 30  # Number of frames to keep in path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for white golf ball (adjust based on lighting)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Track the largest circular contour (likely the ball)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        if radius > 5:  # Filter out small noise
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            ball_positions.append((int(x), int(y)))

    # Draw the ball's path
    for i in range(1, len(ball_positions)):
        cv2.line(frame, ball_positions[i-1], ball_positions[i], (0, 0, 255), 2)

    # Display
    cv2.imshow("Golf Ball Tracker", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()