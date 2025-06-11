import cv2
import os

# Пути к файлам
image_path = "test/1068.jpg"   # Исходное изображение
labels_path = "test/1068.txt"  # Файл с разметкой YOLO
output_path = "test/2.jpg" # Имя файла с наложенной разметкой

# Загрузка изображения
image = cv2.imread(image_path)
height, width, _ = image.shape

# Чтение файла разметки YOLO
with open(labels_path, "r") as file:
    labels = file.readlines()

# Цвет и параметры рамки
color = (255, 0, 0)  # Зеленый
thickness = 10        # Толщина линий

# Обработка каждой строки разметки
for label in labels:
    parts = label.strip().split()
    class_id = int(parts[0])  # Класс объекта (можно использовать, чтобы подписывать)
    x_center, y_center, w, h = map(float, parts[1:])

    # Преобразуем относительные координаты в абсолютные
    x_center, y_center, w, h = (
        int(x_center * width),
        int(y_center * height),
        int(w * width),
        int(h * height),
    )

    # Вычисляем координаты углов bounding box
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    # Рисуем bounding box на изображении
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Добавляем подпись класса
    cv2.putText(image, f"defect", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                2.8, color, 5, cv2.LINE_AA)

# Сохранение нового изображения
cv2.imwrite(output_path, image)
print(f"Изображение с разметкой сохранено как {output_path}")
