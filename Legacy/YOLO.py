import cv2
import time
from ultralytics import YOLO

# Modeli yükle (YOLOv8 veya YOLOv5 otomatik seçilir)
model = YOLO('../demo1/yolov8n.pt')  # veya 'yolov5s.pt' de kullanabilirsin

cap = cv2.VideoCapture(0)
print("Webcam başlatıldı. Çıkmak için 'q' tuşuna bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    results = model(frame)
    elapsed_ms = (time.time() - start_time) * 1000

    # Tespit edilen nesnelerin isimlerini al
    names = results[0].names
    detected = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        detected.append(label)
        # Kutu çiz
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Ekrana yaz
    text = f"Algilananlar: {', '.join(detected) if detected else 'YOK'} | Sure: {elapsed_ms:.1f} ms"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("YOLO Nesne Algılama", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()