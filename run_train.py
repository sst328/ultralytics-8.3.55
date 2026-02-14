from ultralytics import YOLO

def main():
    # 使用 yolov8m 模型
    model = YOLO("yolov8m.pt")

    model.train(
        data="yolov8.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        workers=0,
        project="runs/detect",
        name="train",
    )

if __name__ == "__main__":
    main()