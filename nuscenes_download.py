# nuscenes_download.py
import os
import argparse
from tools.logging_setup import setup_logger
from tools.utils import download_file

DATA_ROOT = "data"
NUSCENES_DIR = os.path.join(DATA_ROOT, "nuscenes")
DOWNLOAD_DIR = os.path.join(NUSCENES_DIR, "downloads")

METADATA_FILES = {
    "v1.0-trainval_meta.tgz": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz",
    "v1.0-mini.tgz": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz",
    "v1.0-test_meta.tgz": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-test_meta.tgz"
}

BLOB_FILES = {
    "v1.0-trainval01_blobs.tgz": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval01_blobs.tgz",
    "v1.0-trainval02_blobs.tgz": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval02_blobs.tgz",
    "v1.0-trainval03_blobs.tgz": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval03_blobs.tgz",
    "v1.0-trainval04_blobs.tgz": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval04_blobs.tgz",
    "v1.0-trainval05_blobs.tgz": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_blobs.tgz",
    "v1.0-trainval06_blobs.tgz": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval06_blobs.tgz",
    "v1.0-trainval07_blobs.tgz": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_blobs.tgz",
    "v1.0-trainval08_blobs.tgz": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval08_blobs.tgz",
    "v1.0-trainval09_blobs.tgz": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval09_blobs.tgz",
    "v1.0-trainval10_blobs.tgz": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval10_blobs.tgz",
    "v1.0-test_blobs.tgz": "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-test_blobs.tgz"
}

def main():
    logger = setup_logger()
    logger.info("🚗 Starting nuScenes Dataset Download...")
    
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    all_files = {**METADATA_FILES, **BLOB_FILES}
    
    for filename, url in all_files.items():
        output_path = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.exists(output_path):
            logger.info(f"✅ {filename} already exists, skipping download.")
        else:
            download_file(url, output_path, f"nuScenes {filename}", logger)
            
    logger.info("✅ nuScenes download process finished.")

if __name__ == "__main__":
    main()