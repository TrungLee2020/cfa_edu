import os
import boto3
from marker.convert import convert_single_pdf
from marker.models import load_all_models

# ===== CONFIG =====
BUCKET_NAME = "bucket-cfa-beaverx"
S3_PREFIX = ""           # Root of the bucket
LOCAL_PDF_DIR = os.path.join(os.getcwd(), "data", "pdfs")
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "ocr_results")

# Marker supports specific languages, usually passed as a list or string depending on version.
# For convert_single_pdf, it often accepts langs as a list.
LANGS = ["vi", "en"]

# ==================

os.makedirs(LOCAL_PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize S3 client. 
# Ensure you have AWS credentials set up in your environment or ~/.aws/credentials
s3 = boto3.client("s3")

# Load models once to avoid reloading for every file
print("Loading Marker models...")
model_lst = load_all_models()

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
        full_text, images, out_meta = convert_single_pdf(
            fname=pdf_path,
            model_lst=model_lst,
            max_pages=None,
            langs=LANGS,
            batch_multiplier=2
        )

        # Save markdown
        md_path = os.path.join(out_dir, "output.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        # Save metadata if needed
        # import json
        # with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        #     json.dump(out_meta, f, indent=4)
            
        print(f"Saved OCR result to {md_path}")

    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")


def main():
    pdfs = download_pdfs()
    
    if not pdfs:
        print("No PDFs found to process.")
        return

    for pdf in pdfs:
        ocr_with_marker(pdf)

    print("DONE OCR ALL FILES")


if __name__ == "__main__":
    main()
