import os
import cv2
from ultralytics import YOLO

# parameters
# path to model to use, change if needed to different model
model_path = '../runs/detect/UrbanWaste_sz800_b16/weights/best.pt'
# path to test images
test_images_path = 'Dataset_test/images'
# output path, change based on which model is being tested
output_dir = '../runs/detect/Top_100_Final_Visuals_sz800_b16'
# for GPU running
runs_on_mac = True

if __name__ == "__main__":
    # make output directory
    os.makedirs(output_dir, exist_ok=True)

    # initialise model with parameters and weights of best previous one
    model = YOLO(model_path)

    # determine correct GPU running settinga
    if runs_on_mac:
        device = 'mps'
    else:
        device = 0

    # use conf=0.01 so we don't miss a single potential piece of trash
    results = list(model.predict(source=test_images_path, imgsz=800,conf=0.01, device=device, verbose=False))

    # extract highest confidence value per image
    image_scores = []
    for r in results:
        if len(r.boxes) > 0:
            # Get the highest confidence value in this specific image
            max_conf = float(r.boxes.conf.max())
            image_scores.append((max_conf, r))

    # sort images on highest confidence first
    image_scores.sort(key=lambda x: x[0], reverse=True)

    # select the top 100 images
    top_100_subset = image_scores[:100]

    # draw all boxes for the selected images
    for _, r in top_100_subset:
        img = cv2.imread(r.path)
        
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > 0.01:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                label = f"{model.names[cls_id]} {conf:.2f}"
                
                # set colours and labels
                color = (0, 255, 0) 
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        filename = os.path.basename(r.path)
        cv2.imwrite(os.path.join(output_dir, filename), img)