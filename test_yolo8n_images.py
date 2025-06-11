import os
from ultralytics import YOLO


def predict_unlabeled_images(
    model_path,
    images_folder,
    output_dir="predictions",
    imgsz=640,
    conf=0.25
):
    """
    Применяет YOLOv8-модель к неразмеченным изображениям из images_folder,
    сжимает каждое до imgsz (внутренне, силами YOLO),
    и сохраняет результаты (картинки с боксами И .txt-файлы в YOLO-формате)
    в output_dir/ (внутри создаётся labels/, где хранятся txt).

    :param model_path: Путь к .pt-файлу (best.pt).
    :param images_folder: Папка с новыми изображениями.
    :param output_dir: Папка, куда сохранять детекции.
    :param imgsz: Размер входа (YOLO resize).
    :param conf: Порог confidence, ниже которого боксы отбрасываются.
    """
    # 1. Загружаем модель
    model = YOLO(model_path)

    # 2. Создаём папку для сохранения результатов, если нужно
    os.makedirs(output_dir, exist_ok=True)

    # 3. Перебираем все файлы в папке images_folder
    for filename in os.listdir(images_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            continue  # пропускаем не-изображения

        img_path = os.path.join(images_folder, filename)

        # 4. Прогоняем через модель
        # Ключевые моменты:
        # - save=True: сохранять отрисованные результаты
        # - save_txt=True: сохранять .txt (YOLO-аннотации)
        # - conf=...: минимальная уверенность
        # - project=output_dir: базовая папка,
        #   внутри будет labels/, где появятся *.txt
        results = model.predict(
            source=img_path,
            imgsz=imgsz,
            conf=conf,
            save=True,
            save_txt=True,
            project=output_dir,
            name="",             # чтобы не создавать вложенную папку exp
            exist_ok=True
        )

    print(f"\nГотово. Результаты сохранены в папку: {output_dir}")
    print("Внутри неё будет subfolder labels/ с *.txt (YOLO формат).")
    print("Теперь можно импортировать изображения + labels в CVAT (YOLO) для редактирования.")


if __name__ == "__main__":
    model_file = "runs/defect/my_defect_yolo8n_default_exp/weights/best.pt"
    new_images = "unlabled_images"   # папка с новыми картинками
    predict_unlabeled_images(
        model_path=model_file,
        images_folder=new_images,
        output_dir="predictions",
        imgsz=640,
        conf=0.25
    )
