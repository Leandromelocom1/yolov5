# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLOv5 modificado: detecção com envio de e-mail e exportação CSV
"""

import argparse
import csv
import os
import smtplib
import ssl
from pathlib import Path
from email.message import EmailMessage

import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements,
    colorstr, cv2, increment_path, non_max_suppression, print_args, scale_boxes
)
from utils.torch_utils import select_device, smart_inference_mode

# CONFIGURAÇÃO DO E-MAIL
EMAIL_FROM = "informatica@engemolde.com.br"
EMAIL_TO = "leandro@engemolde.com.br"
EMAIL_CC = "leandromelo.com@gmail.com"
SMTP_SERVER = "email-ssl.com.br"
EMAIL_USER = EMAIL_FROM
EMAIL_PASS = "25!0521@EnG3819@#"


def enviar_email_com_anexo(imagem_path):
    msg = EmailMessage()
    msg["Subject"] = "Alerta: Pessoa sem EPI detectada"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Cc"] = EMAIL_CC
    msg.set_content("Uma pessoa sem EPI foi detectada. Veja a imagem em anexo.")

    with open(imagem_path, "rb") as img:
        msg.add_attachment(img.read(), maintype="image", subtype="jpeg", filename=os.path.basename(imagem_path))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, 465, context=context) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        print(f"📧 E-mail enviado com sucesso: {imagem_path}")


def write_to_csv(csv_path, image_name, prediction, confidence):
    data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


@smart_inference_mode()
def run(**kwargs):
    opt = argparse.Namespace(**kwargs)
    source = str(opt.source)
    save_img = not opt.nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    (save_dir / "labels" if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn, data=opt.data, fp16=opt.half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(opt.imgsz, s=stride)

    if webcam:
        opt.view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=opt.vid_stride)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=opt.vid_stride)

    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))
    seen, dt = 0, (Profile(device=device), Profile(device=device), Profile(device=device))
    csv_path = save_dir / "predictions.csv"

    for path, im, im0s, _, _ in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device).half() if model.fp16 else torch.from_numpy(im).to(model.device).float()
            im /= 255
            im = im[None] if len(im.shape) == 3 else im

        with dt[1]:
            pred = model(im, augment=opt.augment, visualize=opt.visualize)

        with dt[2]:
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)

        for i, det in enumerate(pred):
            seen += 1
            im0 = im0s[i] if isinstance(im0s, list) else im0s.copy()
            p = Path(path[i] if isinstance(path, list) else path)
            save_path = str(save_dir / p.name)
            imc = im0.copy() if opt.save_crop else im0
            annotator = Annotator(im0, line_width=opt.line_thickness, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if conf < 0.60:
                        continue
                    c = int(cls)
                    label = names[c]
                    confidence_str = f"{float(conf):.2f}"
                    write_to_csv(csv_path, p.name, label, confidence_str)

                    if label.lower() == "sem_epi":
                        crop_dir = save_dir / "detections"
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        full_image_path = crop_dir / f"{p.stem}_sem_epi_full.jpg"
                        cv2.imwrite(str(full_image_path), im0)  # Salva imagem completa
                        try:
                            enviar_email_com_anexo(str(full_image_path))
                        except Exception as e:
                            print(f"Erro ao enviar e-mail com imagem completa: {e}")


                    if save_img:
                        display_label = None if opt.hide_labels else f"{label} {confidence_str}" if not opt.hide_conf else label
                        annotator.box_label(xyxy, display_label, color=colors(c, True))

            im0 = annotator.result()
            if opt.view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
            if save_img:
                cv2.imwrite(save_path, im0)

    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image" % tuple(x.t / seen * 1e3 for x in dt))
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov5s.pt")
    parser.add_argument("--source", type=str, default="data/images")
    parser.add_argument("--data", type=str, default="data/coco128.yaml")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="")
    parser.add_argument("--view-img", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-format", type=int, default=0)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--save-crop", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--classes", nargs="+", type=int)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--line-thickness", default=3, type=int)
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--hide-conf", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dnn", action="store_true")
    parser.add_argument("--vid-stride", type=int, default=1)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements("requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
