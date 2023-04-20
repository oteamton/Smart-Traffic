import cv2

# Load the video
cap = cv2.VideoCapture('D:\\project\\video\\40.mp4')

cap.set(cv2.CAP_PROP_FPS, 30)
# Define the rectangle parameters
x, y = 100, 100    # Top left corner coordinates
w, h = 200, 150    # Width and height
delay = 1/30
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Draw the rectangle on the frame
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
