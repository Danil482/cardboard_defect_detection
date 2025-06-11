import optuna
from ultralytics import YOLO
from optuna.pruners import MedianPruner
from optuna_pruning_callback import create_optuna_callback


def objective(trial: optuna.Trial) -> float:
    """
    Пример с pruner. Обучаем YOLOv8, но передаём колбэк OptunaPruningCallback,
    чтобы при плохих результатах на ранних эпохах trial прерывался.
    """

    # ПАРАМЕТРЫ ДЛЯ ПРИМЕРА
    epochs = trial.suggest_int("epochs", 5, 10)
    batch = trial.suggest_categorical("batch", [4, 8, 16])
    lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
    lrf = trial.suggest_float("lrf", 0.01, 0.2)
    momentum = trial.suggest_float("momentum", 0.85, 0.95)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.001)
    warmup_epochs = trial.suggest_int("warmup_epochs", 1, 3)

    # HSV (аугментации)
    hsv_h = trial.suggest_float("hsv_h", 0.0, 0.02)
    hsv_s = trial.suggest_float("hsv_s", 0.0, 0.9)
    hsv_v = trial.suggest_float("hsv_v", 0.0, 0.5)

    # Перевороты
    flipud = trial.suggest_float("flipud", 0.0, 0.2)
    fliplr = trial.suggest_float("fliplr", 0.0, 0.6)

    # mosaic, mixup, copy_paste
    mosaic = trial.suggest_categorical("mosaic", [0.0, 1.0])
    mixup = trial.suggest_float("mixup", 0.0, 0.3)
    copy_paste = trial.suggest_float("copy_paste", 0.0, 0.2)

    # auto_augment (выкл, randaugment, ...)
    auto_augment = trial.suggest_categorical("auto_augment", ["none", "randaugment"])

    # cos_lr (bool)
    cos_lr = trial.suggest_categorical("cos_lr", [False, True])

    # close_mosaic (int, после скольки эпох отключать mosaic)
    close_mosaic = trial.suggest_int("close_mosaic", 0, 10)

    # deterministic (bool)
    deterministic = trial.suggest_categorical("deterministic", [False, True])

    # Инициализируем модель
    model = YOLO("yolov8n.pt")

    # Создаём колбэк для Optuna
    pruning_callback = create_optuna_callback(trial)

    try:
        model.train(
            data="data.yaml",
            epochs=epochs,
            batch=batch,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            flipud=flipud,
            fliplr=fliplr,
            mosaic=mosaic,
            mixup=mixup,
            copy_paste=copy_paste,
            auto_augment=auto_augment,
            cos_lr=cos_lr,
            close_mosaic=close_mosaic,
            deterministic=deterministic,

            # Доп. параметры
            val=True,  # по умолчанию True, но проверьте
            save_period=1,
            name=f"optuna_trial_{trial.number}",
            project="runs/optuna",
            device=0,
            verbose=False,
            # callbacks=pruning_callback
        )
    except optuna.TrialPruned as e:
        # Если на одной из эпох trial был прерван:
        raise e
    except Exception as e:
        # Любая другая ошибка
        trial.set_user_attr("failed", str(e))
        return 0.0

    # После обучения вызовем model.val() для итоговой метрики
    metrics = model.val()
    map_50_95 = metrics.box.map

    return map_50_95


def main():
    # ИСПОЛЬЗУЕМ MedianPruner (или другой)
    pruner = MedianPruner(n_warmup_steps=5)  # напр. n_warmup_steps=1..2
    study = optuna.create_study(direction="maximize", pruner=pruner)

    # Запускаем оптимизацию
    # При плохих результатах trial будет автоматически прерываться
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    print("=== Итог ===")
    print(f"Лучшая метрика: {study.best_value:.4f}")
    print("Гиперпараметры:")
    for k,v in study.best_trial.params.items():
        print(f"  {k} = {v}")

if __name__ == "__main__":
    main()
