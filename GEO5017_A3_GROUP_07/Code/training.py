from ultralytics import YOLO

# parameters
data_path = 'data.yaml'         # path to data yaml
num_epochs = 30
imgsz = 640
runs_on_mac = True              # for running with GPU

if __name__ == "__main__":
    # load YOLOv8 model
    model = YOLO('yolov8n.pt') 

    # determine correct GPU running settinga
    if runs_on_mac:
        device = 'mps'
    else:
        device = 0

    # train model
    results = model.train(
        data=data_path,
        epochs=num_epochs,
        imgsz=imgsz,           
        device=device,         # for running on GPU
        project='UrbanWaste',
        name='version_1'
    )