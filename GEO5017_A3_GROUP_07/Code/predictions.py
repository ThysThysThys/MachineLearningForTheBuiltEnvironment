import os
import csv
import cv2
from ultralytics import YOLO

# CONFIG
model_path = '../runs/detect/UrbanWaste_yolo11n_bg750_sz800_b16_lr0p001_m1p0_e50/weights/best.pt'
test_images_path = '../Dataset_test/images'
test_labels_path = '../Dataset_test/labels'
output_dir = '../runs/detect/Top_100_Final_Visuals_yolo11n_bg750_sz800_b16_lr0p001_m1p0_e50'

top_k = 100
imgsz = 800
conf_threshold = 0.01
runs_on_mac = True


def contains_waste_ground_truth(image_path, labels_dir):
    """
    Returns True if the image has at least one ground-truth YOLO label.
    Assumes:
      image: some_name.jpg
      label: some_name.txt
    """
    image_name = os.path.basename(image_path)
    stem, _ = os.path.splitext(image_name)
    label_path = os.path.join(labels_dir, stem + ".txt")

    # missing label file -> no annotated waste
    if not os.path.exists(label_path):
        return False

    # non-empty label file -> image contains waste
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    return len(lines) > 0


if __name__ == "__main__":
    # make output directory
    os.makedirs(output_dir, exist_ok=True)

    # initialise model
    model = YOLO(model_path)

    # determine correct device
    if runs_on_mac:
        device = "mps"
    else:
        device = 0  # CUDA GPU on Windows/Linux if available

    # run predictions on all test images
    print("Running predictions on all test images...")
    results = list(
        model.predict(
            source=test_images_path,
            imgsz=imgsz,
            conf=conf_threshold,
            device=device,
            verbose=False
        )
    )

    ranked_images = []

    for r in results:
        if len(r.boxes) > 0:
            max_conf = float(r.boxes.conf.max())
        else:
            max_conf = 0.0

        ranked_images.append({
            "result": r,
            "image_path": r.path,
            "image_name": os.path.basename(r.path),
            "max_conf": max_conf
        })

    # sort descending on max confidence
    ranked_images.sort(key=lambda x: x["max_conf"], reverse=True)

    # select top-k
    top_k_images = ranked_images[:top_k]

    # CALCULATE P@100
    true_waste_count = 0

    for item in top_k_images:
        item["contains_waste_gt"] = contains_waste_ground_truth(
            item["image_path"],
            test_labels_path
        )
        if item["contains_waste_gt"]:
            true_waste_count += 1

    p_at_100 = true_waste_count / top_k

    print("\nRESULTS")
    print("-------")
    print(f"Top {top_k} images selected: {len(top_k_images)}")
    print(f"Images that truly contain waste: {true_waste_count}")
    print(f"P@{top_k}: {p_at_100:.4f}")

    # SAVE TOP-K VISUALIZATIONS
    print("\nSaving top-ranked prediction visualizations...")

    for item in top_k_images:
        r = item["result"]
        img = cv2.imread(r.path)

        for box in r.boxes:
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                label = f"{model.names[cls_id]} {conf:.2f}"

                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                (w, h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )
                cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )

        filename = os.path.basename(r.path)
        cv2.imwrite(os.path.join(output_dir, filename), img)

    print(f"Saved top-{top_k} prediction images to: {output_dir}")

    # SAVE CSV WITH RANKING
    output_csv = os.path.join(output_dir, "top100_p_at_100.csv")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "image_name", "max_conf", "contains_waste_gt"])

        for rank, item in enumerate(top_k_images, start=1):
            writer.writerow([
                rank,
                item["image_name"],
                f"{item['max_conf']:.6f}",
                int(item["contains_waste_gt"])
            ])

    print(f"Saved ranking CSV to: {output_csv}")