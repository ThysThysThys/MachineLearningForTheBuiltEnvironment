from ultralytics import YOLO

# 1. Load a pre-trained "Nano" model (fastest for laptops)
model = YOLO('yolov8n.pt') 

# 2. Start Training
results = model.train(
    data='GEO5017-A3-UrbanWaste/data.yaml',     # Path to your yaml file
    epochs=50,            # How many times to see the data
    imgsz=640,            # Standard image size
    device='mps',         # This uses your Mac's GPU chip!
    project='UrbanWaste', # Name of the folder for results
    name='version_1'
)