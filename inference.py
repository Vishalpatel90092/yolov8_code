from ultralytics import YOLO
model = YOLO("/Users/ravi/runs/detect/Model/predict28.pt")
results = model.predict(source="/Users/ravi/Desktop/data_val", save=True, save_txt=True, project="/Users/ravi/Desktop/data_val", name="predict")


