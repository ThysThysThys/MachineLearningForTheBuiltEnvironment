from ultralytics import YOLO
import os

# 1. Define your parameter grid
# We are testing Resolution, Batch Stability, and Optimizer types
image_sizes = [640, 800]
batch_sizes = [16, 32]
epochs = 30  # 30 is usually enough to see which hyperparameter is winning

# 2. Path to your data
data_path = 'GEO5017-A3-UrbanWaste/data.yaml' # Ensure this uses absolute paths if needed

# 3. Loop through the grid
for imgsz in image_sizes:
    for batch in batch_sizes:
            run_name = f"UrbanWaste_sz{imgsz}_b{batch}"
            print(f"\n🚀 STARTING EXPERIMENT: {run_name}")
            
            # Load a fresh Nano model for each run
            model = YOLO('yolov8n.pt') 
            
            # Train with the current set of hyperparameters
            model.train(
                data=data_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                optimizer='auto',
                device='mps',      # Utilizing your M4 Pro GPU
                name=run_name,     # Saves to runs/detect/run_name
                exist_ok=True,
                mosaic=1.0,        # Keep mosaic high for urban waste
                pretrained=True
            )

print("\n✅ All experiments complete. Check the 'runs/detect/' folder for results.")