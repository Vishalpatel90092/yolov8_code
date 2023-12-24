import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
model_name = "yolov8s"
dataset_name = "/content/drive/MyDrive/DATASET_NOV27_final/Data_training/Data Set/data.yaml" #@param {type:"string"}
model = YOLO(f"{model_name}.pt")
add_wandb_callback(model, enable_model_checkpointing=True)
model.train(project="ultralytics", data=dataset_name, patience=10, epochs=200, imgsz=640)
model.val()
wandb.finish()