import glob
import os
import shutil
from random import shuffle

import cv2

# ----- CONFIG -----
folder_path = input("Enter path to your leaf2 images folder: ")
output_folder = "leaf2-dataset"
train_ratio = 0.8  # 80% train, 20% val
# ------------------

# Clear previous dataset folder if exists
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder + "/images/train", exist_ok=True)
os.makedirs(output_folder + "/images/val", exist_ok=True)
os.makedirs(output_folder + "/labels/train", exist_ok=True)
os.makedirs(output_folder + "/labels/val", exist_ok=True)

# Get all image files
img_extensions = ["*.jpg", "*.jpeg", "*.png"]
images = []
for ext in img_extensions:
    images.extend(glob.glob(os.path.join(folder_path, ext)))

shuffle(images)  # randomize order

train_count = int(len(images) * train_ratio)
train_images = images[:train_count]
val_images = images[train_count:]


def label_images(image_list, dataset_type):
    img_folder = os.path.join(output_folder, "images", dataset_type)
    label_folder = os.path.join(output_folder, "labels", dataset_type)
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    for img_path in image_list:
        img = cv2.imread(img_path)
        if img is None:
            continue
        clone = img.copy()
        boxes = []
        drawing = [False]
        x1y1 = [0, 0]

        def save_yolo_labels(img_name, boxes):
            base = os.path.basename(img_name)
            stem = os.path.splitext(base)[0]
            txt_path = os.path.join(label_folder, stem + ".txt")
            with open(txt_path, "w") as f:
                for box in boxes:
                    xc, yc, w, h = box
                    f.write(f"0 {xc} {yc} {w} {h}\n")  # single class 'leaf2'

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                x1y1[0], x1y1[1] = x, y
                drawing[0] = True
            elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
                temp = img.copy()
                cv2.rectangle(temp, (x1y1[0], x1y1[1]), (x, y), (0, 255, 0), 2)
                cv2.imshow("Image", temp)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing[0] = False
                x2, y2 = x, y
                h_img, w_img, _ = img.shape
                xc = ((x1y1[0] + x2) / 2) / w_img
                yc = ((x1y1[1] + y2) / 2) / h_img
                w_box = abs(x2 - x1y1[0]) / w_img
                h_box = abs(y2 - x1y1[1]) / h_img
                boxes.append((xc, yc, w_box, h_box))
                cv2.rectangle(img, (x1y1[0], x1y1[1]), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("Image", img)

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", mouse_callback)

        print("\n📌 Instructions:")
        print("- Draw box with mouse")
        print("- Press 's' to save and go to next image")
        print("- Press 'r' to reset boxes")
        print("- Press 'q' to quit labeling")

        while True:
            cv2.imshow("Image", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_yolo_labels(img_path, boxes)
                shutil.copy(img_path, img_folder)
                break
            elif key == ord("r"):
                img = clone.copy()
                boxes = []
            elif key == ord("q"):
                print("🚪 Labeling stopped.")
                cv2.destroyAllWindows()
                exit()


# Run labeling
label_images(train_images, "train")
label_images(val_images, "val")

cv2.destroyAllWindows()
print(f"✅ YOLO leaf2 dataset (single class) ready at '{output_folder}'!")
