# Road Visual Detection Pipeline

This project uses the Gemini CLI to automate the detection of anomalies in images. It processes a batch of images, producing bounding boxes and binary segmentation masks for each anomalous object. 

## Features
* **Automated Image Analysis:** Batch processes image folders using `gemini-3-pro-preview` (it works with other models).
* **Model Output:** JSON file containing bounding boxes and polygons, with an additional Python script for visualisation.
* **Resume :** Automatically skips already processed files to prevent extra API calls.

## Prerequisites
* Python 3.8+
* [Gemini CLI](https://github.com/google/generative-ai-python) installed and authenticated.
* Pillow library: `pip install Pillow`

## Setup & Usage

1.  **Configure:**
    Make sure that your *one_anomaly_run_short.json* file and Python script are in the project root. If necessary, update the 'input_folder' path in the script.
    
3.  **Run:**
    ```bash
    python run_visualise_gemini_cli_anomaly.py
    ```

## AI Usage Disclosure
This project uses Google Gemini 3.0 Pro Preview (via the Gemini CLI) as the core reasoning engine for anomaly detection. The model analyses visual data, identifies foreign objects and generates coordinate-based annotations. Although the detection logic relies on AI, the visualisation and binary mask generation are handled deterministically by custom Python scripts.
