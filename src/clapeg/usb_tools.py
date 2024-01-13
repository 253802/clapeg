import csv
import os
import shutil
import time
import torch
import torchvision
import numpy as np
from tkinter import filedialog

import exifread

from PIL import Image
from clapeg.dicom_metadata import generate_report_for_folders, categorize_dicom_by_modality_and_patient

def is_removable_drive(drive):
    """
    Determines whether a given drive is a removable drive.

    Parameters:
    - drive (str): The drive letter to check.

    Returns:
    - bool: True if the drive is removable, False otherwise.
    """
    drive_info = os.popen(f'wmic logicaldisk where caption="{drive}" get drivetype').read()
    return "2" in drive_info


def extract_exif_data(image_path):
    """
    Extracts EXIF metadata from an image file.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - dict: Extracted EXIF data as key-value pairs.
    """
    exif_data = {}
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f)
            for tag, value in tags.items():
                exif_data[tag] = str(value)
    except Exception as e:
        log_error(f"Error extracting EXIF data for {image_path}: {e}")
    return exif_data


def log_error(message):
    """
    Logs an error message with a timestamp to an error log file.

    Parameters:
    - message (str): The error message to log.

    Returns:
    - None
    """
    with open("error_log.txt", "a") as error_log:
        error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def process_image(file_path, save_file_path, csv_writer):
    """
    Processes an image file, copies it to a destination, extracts metadata,
    and writes metadata to a CSV file.

    Parameters:
    - file_path (str): The path to the original image file.
    - save_file_path (str): The path to save the image copy.
    - csv_writer (CSV writer object): The CSV writer object to write metadata.

    Returns:
    - None
    """
    try:
        shutil.copy2(file_path, save_file_path)
        img = Image.open(file_path)
        exif_data = extract_exif_data(file_path)
        metadata = {
            "File": save_file_path,
            "Width": img.width,
            "Height": img.height,
            "Mode": img.mode,
            "Format": img.format,
            "Size": os.path.getsize(file_path),
            "Author": exif_data.get("Image Artist", ""),
            "Device": exif_data.get("Image Model", ""),
            "Exposure": exif_data.get("EXIF ExposureTime", ""),
        }
        csv_writer.writerow(metadata)
        print(f"Image detected and saved: {file_path} -> {save_file_path}")
    except Exception as e:
        log_error(f"Error processing {file_path}: {e}")


def scan_usb_for_images(usb_path, save_path, csv_file):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    try:
        with open(csv_file, 'a', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['File', 'Width', 'Height', 'Mode', 'Format', 'Size', 'Author', 'Device', 'Exposure', 'Object Detected']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Write header if the file is empty
            if os.stat(csv_file.name).st_size == 0:
                writer.writeheader()

            for root, dirs, files in os.walk(usb_path):
                for file in files:
                    _, extension = os.path.splitext(file)
                    if extension.lower() in image_extensions:
                        file_path = os.path.join(root, file)
                        save_file_path = os.path.join(save_path, file)

                        # Copy the image file
                        try:
                            shutil.copy2(file_path, save_file_path)
                        except Exception as e:
                            print(f"Error coping image {file_path}: {e}")

                        # Extract image metadata
                        try:
                            img = Image.open(file_path)
                            _, classes, _, _ = predict(file_path, model, device, 0.9)
                            exif_data = extract_exif_data(file_path)
                            metadata = {
                                'File': save_file_path,
                                'Width': img.width,
                                'Height': img.height,
                                'Mode': img.mode,
                                'Format': img.format,
                                'Size': os.path.getsize(file_path),
                                'Author': exif_data.get('Image Artist', ''),
                                'Device': exif_data.get('Image Model', ''),
                                'Exposure': exif_data.get('EXIF ExposureTime', ''),
                                'Object Detected': classes
                            }
                            writer.writerow(metadata)
                            print(f"Image detected and saved: {file_path} -> {save_file_path}")
                        except Exception as e:
                            print(f"Error getting matadata for image {file_path} or writing into csv file: {e}")

        categorize_dicom_by_modality_and_patient(usb_path, save_path)
        generate_report_for_folders(save_path)
    except Exception as e:
        print(f"Error while opening/creating csv file: {e}")




def detect_plugged_usbs():
    """
    Continuously detects newly plugged removable USB devices
    and initiates scanning for images on them.
    """
    existing_usb_list = []

    result = os.popen("wmic logicaldisk get caption").read()
    current_usb_list = [drive.strip() for drive in result.split("\n") if drive.strip()]

    new_usb_list = [
        drive for drive in current_usb_list if drive not in existing_usb_list and is_removable_drive(drive)
    ]

    for new_usb in new_usb_list:
        print(f"Removable USB Device detected: {new_usb}")
        csv_file_path = f"./metadata_from_{new_usb[0]}.csv"
        scan_usb_for_images(new_usb, f"./images_from_{new_usb[0]}/", csv_file_path)

    existing_usb_list = current_usb_list
    time.sleep(10)


def browse_button(var):
    """
    Opens a file dialog to browse for a directory.

    Parameters:
    - var (Tkinter Variable): A Tkinter variable to set the selected directory.

    Returns:
    - None
    """
    directory = filedialog.askdirectory()
    var.set(directory)

coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval().to(device)

def predict(img, model, device, detection_threshold):
    test = Image.open(img).convert('RGB')
    image = np.array(test)
    image_float_np = np.float32(image) / 255
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    input_tensor = transform(image)
    input_tensor = input_tensor.to(device)
    input_tensor = input_tensor.unsqueeze(0)

    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices