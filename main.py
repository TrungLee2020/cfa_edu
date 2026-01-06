import os
import boto3
from dotenv import load_dotenv

# Set configuration for Marker BEFORE importing or initializing models
# 16GB VRAM is plenty for higher batch sizes
os.environ["INFERENCE_RAM"] = "16"
os.environ["TORCH_DEVICE"] = "cuda"

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import torch
import gc

# Load environment variables (e.g. AWS credentials)
load_dotenv()

# ===== CONFIG =====
BUCKET_NAME = "bucket-cfa-beaverx"
S3_PREFIX = ""           # Root of the bucket
LOCAL_PDF_DIR = os.path.join(os.getcwd(), "data", "pdfs")
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "ocr_results")

# Optimization config
# BATCH_MULTIPLIER removed as it is not supported in converter call


# ==================

if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not found. Marker will run on CPU (Slow).")

os.makedirs(LOCAL_PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

s3 = boto3.client("s3")

# Initialize Marker Converter
print("Loading Marker models...")
# Create config with potential overrides if needed
artifact_dict = create_model_dict()
converter = PdfConverter(
    artifact_dict=artifact_dict,
)

def download_pdfs():
    print(f"Listing files in bucket '{BUCKET_NAME}' with prefix '{S3_PREFIX}'...")
    try:
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=S3_PREFIX
        )
    except Exception as e:
        print(f"Error listing objects: {e}")
        return []

    pdf_files = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                local_path = os.path.join(LOCAL_PDF_DIR, os.path.basename(key))
                if not os.path.exists(local_path):
                    print(f"Downloading {key} to {local_path}...")
                    try:
                        s3.download_file(BUCKET_NAME, key, local_path)
                        pdf_files.append(local_path)
                    except Exception as e:
                        print(f"Error downloading {key}: {e}")
                else:
                    print(f"File {local_path} already exists. Skipping download.")
                    pdf_files.append(local_path)
    else:
        print("No files found in bucket.")
        
    return pdf_files


def ocr_with_marker(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join(OUTPUT_DIR, pdf_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing {pdf_path}...")
    try:
        # Render with optimization params (configured via env vars)
        rendered = converter(pdf_path)
        full_text, _, images = text_from_rendered(rendered)

        # Save images
        for filename, image in images.items():
            image_path = os.path.join(out_dir, filename)
            with open(image_path, "wb") as f:
                f.write(image)

        # Save markdown with the same name as the PDF
        md_path = os.path.join(out_dir, f"{pdf_name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        print(f"Saved OCR result to {md_path}")

    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")


def check_and_clear_vram():
    if not torch.cuda.is_available():
        return

    # Check memory usage
    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_memory = torch.cuda.memory_reserved(0)
    usage_ratio = reserved_memory / total_memory

    # If usage > 66% (approx 2/3), clear cache
    if usage_ratio > 0.66:
        print(f"VRAM usage high ({usage_ratio:.2%}), clearing cache...")
        gc.collect()
        torch.cuda.empty_cache()
        
        # Check again
        new_reserved = torch.cuda.memory_reserved(0)
        print(f"VRAM after clear: {new_reserved / total_memory:.2%}")


def main():
    pdfs = download_pdfs()
    
    if not pdfs:
        print("No PDFs found to process.")
        return

    for i, pdf in enumerate(pdfs):
        ocr_with_marker(pdf)
        
        # Check VRAM after every file
        check_and_clear_vram()

    print("DONE OCR ALL FILES")


if __name__ == "__main__":
    main()
