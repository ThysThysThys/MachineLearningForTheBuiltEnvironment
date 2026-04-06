from ultralytics import YOLO

model_type = "yolo26n.pt"

model = YOLO(model_type)
results = model.train()
