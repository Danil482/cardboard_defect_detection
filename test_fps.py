import time
from ultralytics import YOLO
import cv2

model = YOLO("runs/defect/my_defect_yolo8n_default_exp/weights/best.pt")
# Подгружаем одно тестовое изображение
img = 'dataset_yolo/test/images/0014.jpg'
img = cv2.resize(img, (640, 640))

num_runs = 100
start = time.time()
for _ in range(num_runs):
    results = model(img)
end = time.time()
avg_time = (end - start) / num_runs
fps = 1 / avg_time
print(f"Average inference time per image: {avg_time:.4f} seconds, FPS: {fps:.2f}")
