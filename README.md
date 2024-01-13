# OOD_CL

This project aims to address two primary objectives. The first goal involves developing a script for extracting metadata from DICOM files, enabling the filtering of medical data and enhancing the management of diagnostic images. This effort is aimed at improving the management of medical data and enhancing access to critical diagnostic information within the realm of medical imaging.

The second objective revolves around creating a script for automatic metadata extraction from photographs. This script is designed to optimize the indexing process, facilitate swift retrieval and access to specific images, and find applications across various domains, including the compilation of image collections.

This project is tailored to meet the escalating needs for efficiently managing extensive digital image repositories, catering to the evolving demands in the field of image-centric data management

## Installation steps

1. Create new virtual environment:
    
    ```
    conda create --name clapeg python=3.10
    ```

2. Activate environment
    ```
    conda activate clapeg
    ```

3. Update _pip_ version:
    ```
    python -m pip install --upgrade pip
    ```
4. Install _raunet_ package:

    ```
    python -m pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu117
    ```
5. Enable precommit hook:
    ```
    pre-commit install
    ```