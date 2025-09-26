import os
import shutil
import random
import json
import pandas as pd
from ultralytics import YOLO
import yaml
import time
from pathlib import Path

# Параметры
K = 20  # Для теста используем 1 итерацию, потом можно увеличить до 20
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

IMAGES_DIR = os.path.join("dataset_VOC", "JPEGImages")
LABELS_DIR = "labels"
BOOTSTRAP_LISTS_DIR = "bootstrap_lists"
ORIGINAL_DATA_YAML = "data.yaml"
MODEL_TO_RUN = 'yolov8n_tuned'

models_dict = {
    'yolov8n': {'pt': 'yolov8n.pt', 'tuned': False},
    'yolov8n_tuned': {'pt': 'yolov8n.pt', 'tuned': True},
    'yolov8s': {'pt': 'yolov8s.pt', 'tuned': False},
    'yolov9t': {'pt': 'yolov9t.pt', 'tuned': False},
    'yolov10n': {'pt': 'yolov10n.pt', 'tuned': False},
    'yolo11n': {'pt': 'yolo11n.pt', 'tuned': False},
}


def make_dataset_dirs(base_dataset_dir):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dataset_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_dataset_dir, split, "labels"), exist_ok=True)


def copy_files(file_list, split, base_dir):
    images_subdir = os.path.join(base_dir, split, "images")
    labels_subdir = os.path.join(base_dir, split, "labels")
    copied_labels = 0
    for img_name in file_list:
        src_img_path = os.path.join(IMAGES_DIR, img_name)
        dst_img_path = os.path.join(images_subdir, img_name)
        shutil.copy2(src_img_path, dst_img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_label_path = os.path.join(LABELS_DIR, label_name)
        if os.path.exists(src_label_path):
            dst_label_path = os.path.join(labels_subdir, label_name)
            shutil.copy2(src_label_path, dst_label_path)
            copied_labels += 1
        else:
            print(f"[WARNING] No label for {img_name} (expected {label_name})")
    print(f"Copied {len(file_list)} images, {copied_labels} labels to {split}")


def create_bootstrap_dataset(i, train_files, val_files, test_files):
    bootstrap_dir = f"dataset_yolo_bootstrap_{MODEL_TO_RUN}_{i}"
    make_dataset_dirs(bootstrap_dir)

    print(f"Bootstrap iteration {i + 1}/{K}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    copy_files(train_files, "train", bootstrap_dir)
    copy_files(val_files, "val", bootstrap_dir)
    copy_files(test_files, "test", bootstrap_dir)

    for split in ["train", "val", "test"]:
        images_dir = os.path.join(bootstrap_dir, split, "images")
        labels_dir = os.path.join(bootstrap_dir, split, "labels")
        images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        labels = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
        print(f"{split}: {len(images)} images, {len(labels)} labels")

        # Проверяем валидность .txt файлов
        invalid_labels = []
        for lbl in labels:
            with open(os.path.join(labels_dir, lbl), 'r') as f:
                lines = f.readlines()
                if not lines:
                    invalid_labels.append(lbl)
                else:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            invalid_labels.append(lbl)
                            break
                        try:
                            class_id, x, y, w, h = map(float, parts)
                            if class_id < 0 or x <= 0 or y <= 0 or w <= 0 or h <= 0 or x + w / 2 > 1.0001 or y + h / 2 > 1.0001:
                                invalid_labels.append(lbl)
                                break
                        except ValueError:
                            invalid_labels.append(lbl)
                            break
        if invalid_labels:
            print(f"Обнаружено {len(invalid_labels)} некорректных .txt файлов в {split}: {invalid_labels[:5]}...")

        cache_file = os.path.join(bootstrap_dir, split, "labels.cache")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Removed cache: {cache_file}")

    return bootstrap_dir


def update_data_yaml(bootstrap_dir):
    with open(ORIGINAL_DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)
    base_path = os.path.abspath(bootstrap_dir)
    data['train'] = os.path.join(base_path, 'train/images')
    data['val'] = os.path.join(base_path, 'val/images')
    data['test'] = os.path.join(base_path, 'test/images')
    data['nc'] = 1
    data['names'] = ['defect']
    new_yaml_path = os.path.join(bootstrap_dir, 'data.yaml')
    with open(new_yaml_path, 'w') as f:
        yaml.safe_dump(data, f)
    print(f"Generated data.yaml: {new_yaml_path}")
    with open(new_yaml_path, 'r') as f:
        print(f"data.yaml contents:\n{f.read()}")
    return new_yaml_path


def train_and_eval(model_info, data_yaml_path, i):
    model = YOLO(model_info['pt'])
    name = f"{MODEL_TO_RUN}_bootstrap_{i}"
    project = "runs/defect_bootstrap"

    # Отладка: Проверяем, какие файлы YOLO загружает
    for split in ['train', 'val', 'test']:
        split_path = yaml.safe_load(open(data_yaml_path))[split]
        images = [str(Path(split_path) / f) for f in os.listdir(split_path) if
                  f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"{split} loaded files: {len(images)}")
        print(f"Sample files: {images[:5]}")

    train_params = {
        'data': data_yaml_path,
        'imgsz': 640,
        'name': name,
        'project': project,
        'device': 0,
        'workers': 4,
        'cache': False,
        'plots': False,
        'epochs': 20,
        'batch': 8 if not model_info['tuned'] else 16,
    }
    if model_info['tuned']:
        train_params.update({
            'lr0': 0.0026441230169492085,
            'lrf': 0.10225932922330808,
            'momentum': 0.9389238846531828,
            'weight_decay': 0.0006102966890537429,
            'warmup_epochs': 2,
            'hsv_h': 0.014851249727595078,
            'hsv_s': 0.3802315006699125,
            'hsv_v': 0.4273026490690425,
            'flipud': 0.009430946576148892,
            'fliplr': 0.2831021059215313,
            'mosaic': 1.0,
            'mixup': 0.0254997002852565,
            'copy_paste': 0.05362928257264668,
            'auto_augment': None,
            'cos_lr': True,
            'close_mosaic': 1,
            'deterministic': True,
        })

    start_time = time.time()
    model.train(**train_params)
    end_time = time.time()
    print(f"Training {name} took {end_time - start_time:.2f} seconds")

    best_pt = os.path.join(project, name, "weights/best.pt")
    eval_model = YOLO(best_pt)
    results = eval_model.val(data=data_yaml_path, split="test")

    metrics = {
        'precision': results.results_dict['metrics/precision(B)'],
        'recall': results.results_dict['metrics/recall(B)'],
        'mAP50': results.results_dict['metrics/mAP50(B)'],
        'mAP50-95': results.results_dict['metrics/mAP50-95(B)'],
    }
    return metrics


def main():
    # Очистка предыдущих бутстрэп-директорий
    for i in range(20):
        bootstrap_dir = f"dataset_yolo_bootstrap_{MODEL_TO_RUN}_{i}"
        if os.path.exists(bootstrap_dir):
            shutil.rmtree(bootstrap_dir)
            print(f"Removed previous bootstrap directory: {bootstrap_dir}")

    model_info = models_dict[MODEL_TO_RUN]
    results_list = []

    for i in range(K):
        # Загружаем подвыборку из JSON
        json_path = os.path.join(BOOTSTRAP_LISTS_DIR, f"bootstrap_lists_{i}.json")
        if not os.path.exists(json_path):
            print(f"Ошибка: JSON-файл {json_path} не найден!")
            continue

        with open(json_path, 'r') as f:
            bootstrap_data = json.load(f)

        train_files = bootstrap_data['train']
        val_files = bootstrap_data['val']
        test_files = bootstrap_data['test']

        # Создаём временный датасет
        bootstrap_dir = create_bootstrap_dataset(i, train_files, val_files, test_files)
        data_yaml_path = update_data_yaml(bootstrap_dir)
        metrics = train_and_eval(model_info, data_yaml_path, i)
        metrics['iteration'] = i
        metrics['model'] = MODEL_TO_RUN
        results_list.append(metrics)

        # Удаляем временный датасет после обучения
        shutil.rmtree(bootstrap_dir)
        print(f"Removed temporary bootstrap directory: {bootstrap_dir}")

    df = pd.DataFrame(results_list)
    csv_path = f"bootstrap_results_{MODEL_TO_RUN}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results for {MODEL_TO_RUN} saved to {csv_path}")


if __name__ == '__main__':
    main()
