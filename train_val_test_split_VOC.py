import os
import random


def create_voc_splits(
    voc_root="dataset_VOC",
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    """
    Создаёт train.txt, val.txt, test.txt в папке `VOC_root/ImageSets/Main/`.
    Разбивает изображения из JPEGImages/ на train/val/test в заданных пропорциях.
    """

    # 1. Папки
    images_dir = os.path.join(voc_root, "JPEGImages")
    image_sets_dir = os.path.join(voc_root, "ImageSets", "Main")
    os.makedirs(image_sets_dir, exist_ok=True)

    # 2. Собираем список имён (без расширения)
    image_names = []
    for f in os.listdir(images_dir):
        name, ext = os.path.splitext(f)
        ext = ext.lower()
        # Считаем картинками .jpg/.jpeg/.png
        if ext in [".jpg", ".jpeg", ".png"]:
            image_names.append(name)

    image_names.sort()

    print(f"Всего файлов: {len(image_names)}")
    if len(image_names) == 0:
        print("Нет изображений, выходим.")
        return

    # 3. Перемешиваем и делим
    random.seed(seed)
    random.shuffle(image_names)

    total_count = len(image_names)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count

    train_split = image_names[:train_count]
    val_split = image_names[train_count:train_count+val_count]
    test_split = image_names[train_count+val_count:]

    print(f"Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")

    # 4. Записываем txt-файлы
    with open(os.path.join(image_sets_dir, "train.txt"), 'w') as f:
        for name in train_split:
            f.write(name + "\n")

    with open(os.path.join(image_sets_dir, "val.txt"), 'w') as f:
        for name in val_split:
            f.write(name + "\n")

    with open(os.path.join(image_sets_dir, "test.txt"), 'w') as f:
        for name in test_split:
            f.write(name + "\n")

    print(f"Файлы train.txt, val.txt, test.txt созданы в {image_sets_dir}.")


if __name__ == "__main__":
    create_voc_splits(
        voc_root="dataset_VOC",  # при необходимости укажите другой путь
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
