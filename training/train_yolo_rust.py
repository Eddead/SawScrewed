import os
from ultralytics import YOLO

def train_model(model_path, model_name):
    model = YOLO(model_path)
    model.train(
        data='data.yaml',
        epochs=500,
        imgsz=640,
        batch=32,
        cache=False,
        name=f'Screw500E640Size{model_name}',      # Custom run name
        augment=True,
        device='cuda:0', # change to cpu if no gpu available
        project='assignmentModels',
        # Augmentations
        close_mosaic=50,
        hsv_h=0.05,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.2,
        scale=0.3,
        shear=2.0,
        flipud=0.2,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.3,
        patience = 0,

        # Training optimizations
        cos_lr=True,
        warmup_epochs=3
    )

def main():
    base_path = "model_used/nano" # base models folder
    # model_versions = ["yolov10n.pt", "yolov9t.pt", ] train YOLO models in batches
    model_versions = ["yolov8n.pt"]

    for model_file in model_versions:
        model_path = os.path.join(base_path, model_file)
        model_name = os.path.splitext(model_file)[0].capitalize()  # e.g., 'Yolo11n'
        print(f"\nTraining: Screw{model_name}")
        train_model(model_path, model_name)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
