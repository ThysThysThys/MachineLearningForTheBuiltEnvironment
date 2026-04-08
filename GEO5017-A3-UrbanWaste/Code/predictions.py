import os
import cv2
from ultralytics import YOLO

# 1. Setup
model_path = 'runs/detect/UrbanWaste_sz800_b32/weights/best.pt'
test_images_path = 'GEO5017-A3-UrbanWaste/Dataset_test/images'
output_dir = 'runs/detect/Top_100_Final_Visuals_sz800_b32'
os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_path)

# 2. Run prediction with a VERY low threshold to find everything
# We use conf=0.01 so we don't miss a single potential piece of trash
print("🔍 Scanning all test images...")
results = list(model.predict(source=test_images_path, conf=0.01, device='mps', verbose=False))

# 3. Rank images by their BEST detection
image_scores = []
for r in results:
    if len(r.boxes) > 0:
        # Get the highest confidence value in this specific image
        max_conf = float(r.boxes.conf.max())
        image_scores.append((max_conf, r))

# Sort images: highest confidence first
image_scores.sort(key=lambda x: x[0], reverse=True)

# Select the top 100 images
top_100_subset = image_scores[:100]

print(f"🖋️ Drawing all detections on the best {len(top_100_subset)} images...")

# 4. Draw all boxes for the selected images
for _, r in top_100_subset:
    img = cv2.imread(r.path)
    
    # Draw every box the model found in this image
    for box in r.boxes:
        conf = float(box.conf[0])
        # We only draw boxes with at least 15% confidence to keep it clean
        if conf > 0.01:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            
            # Neon Green for high precision look
            color = (0, 255, 0) 
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    filename = os.path.basename(r.path)
    cv2.imwrite(os.path.join(output_dir, filename), img)

print(f"✅ DONE! Check {output_dir} for exactly {len(top_100_subset)} images.")