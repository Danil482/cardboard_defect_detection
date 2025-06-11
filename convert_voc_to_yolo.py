import os
import xml.etree.ElementTree as ET
import cv2


# Папки (укажите свои пути):
IMAGES_DIR = r"C:\Users\dania\PycharmProjects\Master work\JPEGImages"  # папка с изображениями
ANNOTATIONS_DIR = r"C:\Users\dania\PycharmProjects\Master work\Annotations"  # папка с XML
LABELS_DIR = r"C:\Users\dania\PycharmProjects\Master work\labels"  # папка для выходных .txt

os.makedirs(LABELS_DIR, exist_ok=True)

# Глобальный словарь: класс -> индекс
# Если вы уже знаете несколько классов, можете добавить их сюда заранее:
# CLASSES = {"defect": 0, "scratch": 1}
# А можно начать с пустого словаря:
CLASSES = {}


def convert_voc_to_yolo(xml_file_path):
    """
    Читает один XML-файл в формате Pascal VOC, возвращает (filename, [строки YOLO]).
    Строки YOLO: <class_index> <x_center> <y_center> <width> <height> (все координаты [0..1]).
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    filename = root.find('filename').text  # например, '0001.jpg'
    image_path = os.path.join(IMAGES_DIR, filename)

    # Попробуем считать картинку
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARNING] Can't open image {image_path}. Skipping.")
        return filename, []

    img_h, img_w, _ = img.shape

    yolo_lines = []
    # Проходим по всем <object> в XML
    for obj in root.findall('object'):
        cls_name = obj.find('name').text.strip()  # название класса

        # Если класс не встречался - динамически добавляем
        if cls_name not in CLASSES:
            cls_idx = len(CLASSES)  # индекс = текущий размер словаря
            CLASSES[cls_name] = cls_idx
            print(f"[INFO] Added new class '{cls_name}' with id={cls_idx}")
        else:
            cls_idx = CLASSES[cls_name]

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # Переводим в YOLO-формат (нормализуем)
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        bbox_width = (xmax - xmin) / img_w
        bbox_height = (ymax - ymin) / img_h

        yolo_line = f"{cls_idx} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
        yolo_lines.append(yolo_line)

    return filename, yolo_lines


def main():
    # Обойдём все XML-файлы в папке ANNOTATIONS_DIR
    for xml_file in os.listdir(ANNOTATIONS_DIR):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
        filename, yolo_boxes = convert_voc_to_yolo(xml_path)

        # Проверим, существует ли соответствующее изображение
        image_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.isfile(image_path):
            print(f"[WARNING] image file {filename} not found in {IMAGES_DIR}. Skipping.")
            continue

        # Создадим txt-файл с таким же именем, как у картинки
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(LABELS_DIR, txt_filename)

        with open(txt_path, 'w') as f:
            for line in yolo_boxes:
                f.write(line + "\n")

        print(f"[OK] {xml_file} -> {txt_filename} [{len(yolo_boxes)} bboxes processed]")

    # После обработки всех файлов можно вывести/сохранить полученный словарь классов
    print("\n=== Итоговый список классов ===")
    # Сортируем по индексу: (class_name, class_idx)
    sorted_classes = sorted(CLASSES.items(), key=lambda x: x[1])
    for cls_name, cls_idx in sorted_classes:
        print(f"{cls_idx}: {cls_name}")

    # По желанию можно сохранить словарь в текстовый файл classes.txt
    # с классами по строкам (используется, например, в data.yaml YOLOv5/8).
    # Пример:
    classes_txt_path = os.path.join(LABELS_DIR, "classes.txt")
    with open(classes_txt_path, 'w') as f:
        for cls_name, cls_idx in sorted_classes:
            f.write(f"{cls_name}\n")
    print(f"\nСписок классов сохранён в {classes_txt_path}")


if __name__ == "__main__":
    main()
