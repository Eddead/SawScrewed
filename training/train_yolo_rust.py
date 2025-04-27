from ultralytics import YOLO

def main():
    model_path = ()
    model = YOLO(model_path)

    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=960,
        batch=8,
        name='screw_rust_detection',
        augment=True,
        device='cuda:0',  # or 'cpu' if you want
        project='trainedModels'
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Optional but good practice on Windows
    main()
