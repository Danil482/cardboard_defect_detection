import os
import shutil
import random


IMAGES_DIR = os.path.join("dataset_VOC/JPEGImages")
LABELS_DIR = os.path.join("labels")

# Папка, куда хотим собрать датасет в новой структуре
DATASET_DIR = "dataset_yolo"

# Соотношение для разбиения
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Для воспроизводимости
random.seed(42)


def make_dataset_dirs(base_dataset_dir):
    """
    Создаёт структуру:
      dataset_yolo/
        train/images
        train/labels
        val/images
        val/labels
        test/images
        test/labels
    """
    for split in ["train", "val", "test"]:
        images_subdir = os.path.join(base_dataset_dir, split, "images")
        labels_subdir = os.path.join(base_dataset_dir, split, "labels")
        os.makedirs(images_subdir, exist_ok=True)
        os.makedirs(labels_subdir, exist_ok=True)


def main():
    # 1. Соберём список всех файлов изображений
    all_images = [f for f in os.listdir(IMAGES_DIR)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Всего изображений: {len(all_images)}")

    # 2. Тасуем и делим
    random.shuffle(all_images)
    total_count = len(all_images)
    train_count = int(total_count * TRAIN_RATIO)
    val_count = int(total_count * VAL_RATIO)
    test_count = total_count - train_count - val_count

    train_files = all_images[:train_count]
    val_files = all_images[train_count:train_count + val_count]
    test_files = all_images[train_count + val_count:]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # 3. Создаём новую структуру dataset_yolo/
    make_dataset_dirs(DATASET_DIR)

    # 4. Функция копирования
    def copy_files(file_list, split):
        """
        Копирует изображения + аннотации в:
          dataset_yolo/<split>/images/
          dataset_yolo/<split>/labels/
        """
        images_subdir = os.path.join(DATASET_DIR, split, "images")
        labels_subdir = os.path.join(DATASET_DIR, split, "labels")

        for img_name in file_list:
            # Исходный путь к картинке
            src_img_path = os.path.join(IMAGES_DIR, img_name)
            # Целевой путь
            dst_img_path = os.path.join(images_subdir, img_name)

            # Копируем или переносим (здесь copy2, если нужен именно move, замените)
            shutil.copy2(src_img_path, dst_img_path)

            # Ищем аннотацию
            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_label_path = os.path.join(LABELS_DIR, label_name)
            if os.path.exists(src_label_path):
                dst_label_path = os.path.join(labels_subdir, label_name)
                shutil.copy2(src_label_path, dst_label_path)
            else:
                print(f"[WARNING] Нет аннотации для {img_name} (ожидается {label_name})")

    # 5. Копируем файлы
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print("\nРазбиение завершено! Итоговая структура в папке:", DATASET_DIR)


if __name__ == "__main__":
    main()
