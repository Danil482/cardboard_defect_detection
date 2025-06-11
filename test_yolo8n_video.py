import os
from ultralytics import YOLO

def predict_unlabeled(
    model_path,
    source_path,
    output_dir="predictions",
    imgsz=640,
):
    """
    Применяет YOLOv8-модель к неразмеченным данным (изображения/видео).
    - Если source_path - это папка, то обрабатываем все файлы в ней.
    - Если source_path - это одиночный файл (картинка или видео),
      то обрабатываем только его.

    :param model_path: Путь к .pt-файлу (best.pt).
    :param source_path: Путь к папке или единственному файлу.
    :param output_dir: Папка, куда сохранять детекции.
    :param imgsz: Размер входа (YOLO resize).
    """

    # 1. Загружаем модель
    model = YOLO(model_path)
    # 2. Создаём папку для сохранения результатов, если нужно
    os.makedirs(output_dir, exist_ok=True)

    # Разрешённые расширения для изображений
    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    # Разрешённые расширения для видео
    video_exts = [".mp4", ".avi", ".mov", ".mkv"]

    # Проверяем, папка или файл
    if os.path.isdir(source_path):
        # Если папка, то обходим все файлы
        for filename in os.listdir(source_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_exts or ext in video_exts:
                file_path = os.path.join(source_path, filename)
                model.predict(
                    source=file_path,
                    imgsz=imgsz,
                    save=True,
                    project=output_dir,
                    name="",
                    exist_ok=True,
                )
    else:
        # Если одиночный файл
        ext = os.path.splitext(source_path)[1].lower()
        if ext in image_exts or ext in video_exts:
            model.predict(
                source=source_path,
                imgsz=imgsz,
                save=True,
                project=output_dir,
                name="",
                exist_ok=True,
            )
        else:
            print(f"Файл {source_path} не является поддерживаемым форматом (изображение/видео).")

    print(f"Готово. Результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    model_file = "runs/defect/my_defect_yolo8n_default_exp/weights/best.pt"
    # Пример: либо указываем папку, где .jpg/.png/.mp4
    # либо один файл ( .mp4 или .jpg )
    source_path = "video/0001.mp4"  # пример

    predict_unlabeled(
        model_path=model_file,
        source_path=source_path,
        output_dir="predictions",
        imgsz=640
    )
