import cv2
from fer import FER

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create FER emotion detector
emotion_detector = FER(mtcnn=True)  # mtcnn=True makes face detection more accurate

print("🚀 Starting emotion detection... Press 'q' to quit.")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions in the frame
    emotions = emotion_detector.detect_emotions(frame)

    # Draw boxes and emotion labels
    for e in emotions:
        (x, y, w, h) = e["box"]
        emotion, score = max(e["emotions"].items(), key=lambda item: item[1])

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Display emotion label
        cv2.putText(frame, f"{emotion} ({score:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Emotion Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
