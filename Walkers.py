import cv2

# Load the classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Open the video file
cap = cv2.VideoCapture('walking.avi')

# Loop through the frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Stop the loop if no frame is read
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bodies in the frame
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    # Draw rectangles around the detected bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with detected bodies
    cv2.imshow('Detected Bodies', frame)

    # Wait for a key event and exit the program if 'q' is pressed
    if cv2.waitKey(25) == (32):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
