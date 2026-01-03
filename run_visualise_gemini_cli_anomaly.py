import subprocess
import json
import os
import time
from pathlib import Path
from PIL import Image, ImageDraw

# --- Configuration ---
input_folder = "./dataset_ObstacleTrack/test"    # Where your source images are
output_folder = "./dataset_ObstacleTrack/test_output" # Where JSON & Annotations will go
prompt_file = "one_anomaly_run_short.json"       # Your JSON prompt file
allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
cli_model = "gemini-3-pro-preview"
# ---------------------

def visualize_result(image_path, json_path):
    """
    Reads the original image and the generated JSON, 
    draws annotations, and saves `_annotated.jpg` to the OUTPUT folder.
    """
    try:
        # Load Data from the Output Folder
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Open Original Image from Input Folder
        img = Image.open(image_path).convert("RGBA")
        width, height = img.size
        
        # Create transparent overlay
        poly_overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw_poly = ImageDraw.Draw(poly_overlay)
        draw_lines = ImageDraw.Draw(img)
        
        anomalies = data.get("anomalies", [])
        if not anomalies:
            return 

        for item in anomalies:
            # 1. Draw Polygon Mask
            if "polygon_points" in item and len(item['polygon_points']) > 2:
                pixel_points = [(p[0] * width, p[1] * height) for p in item['polygon_points']]
                draw_poly.polygon(pixel_points, fill=(255, 255, 0, 100))
                draw_lines.polygon(pixel_points, outline="#FFD700")

            # 2. Draw Bounding Box
            if "box_2d" in item:
                ymin, xmin, ymax, xmax = item['box_2d']
                box = [xmin * width, ymin * height, xmax * width, ymax * height]
                draw_lines.rectangle(box, outline="red", width=3)
        
        # Composite
        final_img = Image.alpha_composite(img, poly_overlay)
        final_img = final_img.convert("RGB")
        
        # SAVE to the same directory as the JSON (The Output Folder)
        output_vis_path = json_path.parent / f"{image_path.stem}_annotated.jpg"
        final_img.save(output_vis_path)
        print(f"    [Visual] Saved annotation: {output_vis_path.name}")

    except Exception as e:
        print(f"    [Visual Error] Could not visualize {image_path.name}: {e}")

def process_image(image_path, prompt_text, out_dir):
    """Runs CLI and saves results to the specified output directory."""
    
    # Construct output path: output_folder/image_name.json
    json_output_path = out_dir / image_path.with_suffix(".json").name
    
    # Resume Logic
    if json_output_path.exists():
        print(f"Skipping {image_path.name} (JSON already exists in output)")
        # Check if visualization is missing, if so, regenerate it
        vis_path = out_dir / f"{image_path.stem}_annotated.jpg"
        if not vis_path.exists():
             visualize_result(image_path, json_output_path)
        return

    print(f"Analyzing: {image_path.name}...")

    # Combine prompt and image for CLI
    full_instruction = f"{prompt_text} @{str(image_path)}"
    
    command = [
        "gemini", 
        "-m", cli_model,
        "-s",
        full_instruction
    ]

    try:
        # Run Gemini CLI
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        
        # Save JSON to Output Folder
        with open(json_output_path, "w") as f:
            f.write(result.stdout)
        
        # Visualize
        if result.stdout.strip().startswith("{"):
            print(f"    [CLI] JSON Saved.")
            visualize_result(image_path, json_output_path)
        else:
            print(f"    [CLI Warning] Output might not be valid JSON.")

    except subprocess.CalledProcessError as e:
        print(f"    [CLI Error] Failed to process: {e.stderr}")

def main():
    # 1. Validation
    if not os.path.exists(prompt_file):
        print(f"Critical Error: Prompt file '{prompt_file}' not found.")
        return

    in_path = Path(input_folder)
    if not in_path.exists():
        print(f"Critical Error: Input folder '{input_folder}' not found.")
        return

    # 2. Create Output Folder if it doesn't exist
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {out_path.resolve()}")

    # 3. Load Prompt
    with open(prompt_file, "r") as f:
        prompt_text = f.read()

    # 4. Find Images
    images = [
        p for p in in_path.iterdir() 
        if p.suffix.lower() in allowed_extensions
    ]
    
    print(f"Found {len(images)} images to process.")
    print("-" * 40)

    # 5. Processing Loop
    for i, img in enumerate(images):
        process_image(img, prompt_text, out_path)
        time.sleep(1) 

if __name__ == "__main__":
    main()