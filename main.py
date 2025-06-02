import cv2
import numpy as np

# Load video (replace with 0 for webcam)
cap = cv2.VideoCapture("vivaanswing.mp4")

# Tracking variables
ball_positions = []
club_positions = []
trajectory_length = 30  # Number of frames to keep in path

# Color ranges (adjust these based on your actual ball and club colors)
# For white golf ball
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

# For clubhead (example for a dark-colored club - adjust as needed)
lower_club = np.array([0, 50, 0])
upper_club = np.array([180, 255, 100])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Ball detection
    ball_mask = cv2.inRange(hsv, lower_white, upper_white)
    ball_contours, _ = cv2.findContours(ball_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Club detection
    club_mask = cv2.inRange(hsv, lower_club, upper_club)
    club_contours, _ = cv2.findContours(club_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Track the largest circular contour (likely the ball)
    if ball_contours:
        largest_ball = max(ball_contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_ball)
        
        if radius > 5:  # Filter out small noise
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            ball_positions.append((int(x), int(y)))

    # Track the clubhead (looking for a non-circular shape)
    if club_contours:
        # Find the contour with significant area but not too circular
        club_contours = sorted(club_contours, key=cv2.contourArea, reverse=True)[:3]
        
        for contour in club_contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold for clubhead
                # Get the centroid of the clubhead
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                    club_positions.append((cx, cy))
                    break  # Only track the largest valid club contour

    # Draw the ball's path (green)
    for i in range(1, len(ball_positions)):
        if i < len(ball_positions):
            cv2.line(frame, ball_positions[i-1], ball_positions[i], (0, 255, 0), 2)

    # Draw the club's path (blue)
    for i in range(1, len(club_positions)):
        if i < len(club_positions):
            cv2.line(frame, club_positions[i-1], club_positions[i], (255, 0, 0), 2)

    # Limit trajectory length
    if len(ball_positions) > trajectory_length:
        ball_positions.pop(0)
    if len(club_positions) > trajectory_length:
        club_positions.pop(0)

    # Display
    cv2.imshow("Golf Swing Tracker", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()