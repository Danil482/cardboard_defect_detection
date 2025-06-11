from mmengine import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules


def main():
    register_all_modules()

    # 1. Читаем файл конфигурации (это не просто строка, а полноценный объект Config)
    config_file = '.venv/Lib/site-packages/mmdet/.mim/configs/ssd/ssd300_coco.py'
    cfg = Config.fromfile(config_file)

    # 2. Создаём Runner на основе cfg
    runner = Runner.from_cfg(cfg)

    # 3. Запуск обучения
    runner.train()


if __name__ == '__main__':
    main()
