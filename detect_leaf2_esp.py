import requests
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    check_file,
    check_img_size,
    check_imshow,
    cv2,
    increment_path,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
    strip_optimizer,
    colorstr
)
from utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils.plotting import Annotator, colors

# ESP32 configuration
ESP32_IP = "192.168.108.54"   # replace with your ESP32 IP
ESP32_ON = f"{ESP32_IP}/motor/on"
ESP32_OFF = f"{ESP32_IP}/motor/off"


@smart_inference_mode()
def run(
    weights=ROOT / "leaf2_model2/weights/best.pt",
    source=0,  # webcam
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=True,
    classes=None,
    agnostic_nms=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):
    # Directories
    save_dir = increment_path(Path(ROOT) / "runs/detect/leaf2_esp", exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Load source
    webcam = str(source).isdigit()
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        # Inference
        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            im0 = im0s[i].copy() if webcam else im0s
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c]
                    confidence = float(conf)

                    # Draw boxes
                    if not hide_labels:
                        annotator.box_label(xyxy, f"{label} {confidence:.2f}", color=colors(c, True))

                    # 🚨 Trigger ESP32 if confidence > 0.7
                    if label == "leaf2" and confidence > 0.7:
                        try:
                            requests.get(ESP32_ON, timeout=1)
                            print(f"✅ Motor ON (leaf2 detected, conf={confidence:.2f})")
                        except:
                            print("⚠️ ESP32 connection failed")
                    else:
                        try:
                            requests.get(ESP32_OFF, timeout=1)
                            print("⛔ Motor OFF")
                        except:
                            pass

            # Show results
            if view_img:
                cv2.imshow(str(path), annotator.result())
                if cv2.waitKey(1) == ord("q"):
                    raise StopIteration


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "leaf2_model2/weights/best.pt", help="model path")
    parser.add_argument("--source", type=str, default="0", help="source (0 for webcam)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU threshold")
    parser.add_argument("--device", default="", help="cuda device or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
