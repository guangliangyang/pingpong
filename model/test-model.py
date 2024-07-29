
# 在视频上进行推理
import cv2
from ultralytics import YOLOv10

model = YOLOv10('pingpong_table_best10.pt')  # 加载训练好的模型
cap = cv2.VideoCapture('table.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 推理
    results = model.predict(frame)

    print(f"Number of table results: {len(results)}")
    # 绘制检测框
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 显示结果
    cv2.imshow('YOLOv10', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()