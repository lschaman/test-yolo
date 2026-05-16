from ultralytics import YOLO
import cv2

model = YOLO('../yolo-weight/yolov8n.pt')
results = model("image/new.jpg", show=True)
img = results[0].plot()
cv2.imshow("YOLO Result", img)
cv2.waitKey(0)