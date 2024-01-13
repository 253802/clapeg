import logging
import os
import shutil

from typing import (
    Any,
    Dict,
    Union,
)

import numpy as np
import pydicom

from clapeg.config import KEYS
from clapeg.dicom_stats import extract_image_statistics


def extract_metadata(image_data: pydicom.dataset.FileDataset) -> Dict[str, Any]:
    """
    Extracts metadata from a DICOM dataset.

    Args:
    - image_data (pydicom.dataset.FileDataset): DICOM dataset.

    Returns:
    - Dict[str, Any]: Dictionary containing extracted metadata.
    """
    metadata = {}
    for key, dicom_key in KEYS.items():
        try:
            value = getattr(image_data, dicom_key, "Unknown")
            metadata[key] = value
        except Exception as e:
            logging.warning(f"Error extracting {key}: {e}")
            metadata[key] = "Unknown"
    return metadata


def extract_dicom_meta_data(filename: str) -> Union[Dict[str, Union[str, Any]], None]:
    try:
        image_data = pydicom.read_file(filename)
        img = np.array(image_data.pixel_array).flatten()

        metadata = extract_metadata(image_data)
        image_stats = extract_image_statistics(img)

        return {**metadata, **image_stats}

    except FileNotFoundError as e:
        logging.error(f"File not found for {filename} - {e}")
    except pydicom.errors.InvalidDicomError as e:
        logging.error(f"Invalid DICOM file {filename} - {e}")
    except Exception as e:
        logging.error(f"An error occurred while processing {filename} - {e}")

    return None


def categorize_dicom_by_modality_and_patient(source_folder: str, output_folder: str) -> None:
    """
    Categorizes DICOM files from a source folder based on their modality value
    and patient ID, and moves them into respective output folders.

    Args:
    - source_folder (str): Path to the folder containing DICOM files.
    - output_folder (str): Path to the output folder to organize the files.
    """
    try:
        os.makedirs(output_folder, exist_ok=True)

        for root, _, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".dcm"):
                    try:
                        image_data = pydicom.read_file(file_path)
                        modality = getattr(image_data, KEYS["Modality"], "Unknown")
                        patient_id = getattr(image_data, KEYS["Patient"], "Unknown")

                        modality_folder = os.path.join(output_folder, modality)
                        os.makedirs(modality_folder, exist_ok=True)

                        patient_folder = os.path.join(modality_folder, patient_id)
                        os.makedirs(patient_folder, exist_ok=True)

                        shutil.copy2(file_path, os.path.join(patient_folder, file))

                    except Exception as e:
                        logging.warning(f"Error processing {file_path}: {e}")

    except Exception as e:
        logging.error(f"An error occurred while categorizing DICOM files: {e}")


def generate_report_for_folders(output_folder: str) -> None:
    """
    Generates a report for each folder in the output_folder with extracted metadata
    from a random DICOM file within that folder.

    Args:
    - output_folder (str): Path to the output folder containing categorized DICOM files.
    """
    try:
        for root, dirs, _ in os.walk(output_folder):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
                if files:
                    random_file = np.random.choice(files)
                    random_file_path = os.path.join(folder_path, random_file)
                    try:
                        image_data = pydicom.read_file(random_file_path)
                        metadata = extract_metadata(image_data)
                        report_file = os.path.join(folder_path, f"{folder}_report.txt")
                        with open(report_file, "w") as report:
                            report.write("Metadata for a random DICOM file:\n")
                            for key, value in metadata.items():
                                report.write(f"{key}: {value}\n")
                    except Exception as e:
                        logging.warning(f"Error processing {random_file_path}: {e}")
    except Exception as e:
        logging.error(f"An error occurred while generating reports: {e}")
