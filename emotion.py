import cv2
from fer import FER


cap = cv2.VideoCapture(0)


emotion_detector = FER(mtcnn=True)  

print("🚀 Starting emotion detection... Press 'q' to quit.")

while True:
   
    ret, frame = cap.read()
    if not ret:
        break

    emotions = emotion_detector.detect_emotions(frame)


    for e in emotions:
        (x, y, w, h) = e["box"]
        emotion, score = max(e["emotions"].items(), key=lambda item: item[1])

      
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(frame, f"{emotion} ({score:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

  
    cv2.imshow("Emotion Detection", frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

