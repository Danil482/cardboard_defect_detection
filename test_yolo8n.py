from ultralytics import YOLO


def test_model():
    # 1. Путь к обученной модели (best.pt)
    model_path = "runs/defect/my_defect_yolo8n_default_exp/weights/best.pt"

    # 2. Загружаем модель
    model = YOLO(model_path)

    # 3. Выполняем валидацию на тестовых данных
    #    (если в data.yaml есть test: ..., то можно split='test')
    results = model.val(
        data="data.yaml",  # data.yaml с путём к test-набору
        split="test"       # указывает использовать test-выборку из data.yaml
    )


if __name__ == "__main__":
    test_model()
