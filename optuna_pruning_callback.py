import optuna
from ultralytics import YOLO


def create_optuna_callback(trial):
    def on_val_end(trainer):
        # Вал приходит в конце каждой эпохи, если YOLO настроен на val
        epoch_idx = trainer.epoch
        # Посмотрим, что в trainer.metrics
        print("on_val_end metrics:", trainer.metrics)
        map_50_95 = trainer.metrics.get('box/map50-95', None)
        if map_50_95 is not None:
            # Пропустим первые 2 эпохи для безопасности
            if epoch_idx >= 2:
                trial.report(map_50_95, step=epoch_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned on epoch {epoch_idx}")

    return {
        "on_val_end": on_val_end
    }

