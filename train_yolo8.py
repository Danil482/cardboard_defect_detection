from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        name="my_defect_yolov8n_default_exp2",
        project="runs/defect",
        device=0
    )
    
    # model.train(
    #     data="data.yaml",
    #     epochs=100,
    #     batch=16,
    #     imgsz=640,
    #     name="my_defect_yolo8n_tunned_exp",
    #     project="runs/defect",
    #     device=0,
    #     lr0=0.0026441230169492085,
    #     lrf=0.10225932922330808,
    #     momentum=0.9389238846531828,
    #     weight_decay=0.0006102966890537429,
    #     warmup_epochs=2,
    #     hsv_h=0.014851249727595078,
    #     hsv_s=0.3802315006699125,
    #     hsv_v=0.4273026490690425,
    #     flipud=0.009430946576148892,
    #     fliplr=0.2831021059215313,
    #     mosaic=1.0,
    #     mixup=0.0254997002852565,
    #     copy_paste=0.05362928257264668,
    #     auto_augment=None,
    #     cos_lr=True,
    #     close_mosaic=1,
    #     deterministic=True,
    # )


if __name__ == "__main__":
    main()
