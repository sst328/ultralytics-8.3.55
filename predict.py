from ultralytics import YOLO

MODEL_WEIGHTS = "runs/detect/train/weights/best.pt"
SOURCE = "test_imgs"

def main():
    model = YOLO(MODEL_WEIGHTS)
    results = model.predict(source=SOURCE, save=True, conf=0.25, batch=8)
    print("预测完成，结果在 runs/detect/predict 下。")

if __name__ == "__main__":
    main()