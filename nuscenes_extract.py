# nuscenes_extract.py
import os
import subprocess
from tools.logging_setup import setup_logger
from tools.utils import check_pigz_available, extract_archive, check_mmdet3d_available

DATA_ROOT = "data"
NUSCENES_DIR = os.path.join(DATA_ROOT, "nuscenes")
DOWNLOAD_DIR = os.path.join(NUSCENES_DIR, "downloads")

def main():
    logger = setup_logger()
    logger.info("🚗 Starting nuScenes Dataset Extraction and Setup...")

    if not os.path.exists(DOWNLOAD_DIR) or not os.listdir(DOWNLOAD_DIR):
        logger.error("❌ Download directory is empty. Please run 'nuscenes_download.py' first.")
        return

    use_pigz = check_pigz_available(logger)

    # Extract all .tgz files
    for filename in os.listdir(DOWNLOAD_DIR):
        if filename.endswith(('.tgz', '.tar.gz')):
            archive_path = os.path.join(DOWNLOAD_DIR, filename)
            # Simple check: if a folder with a similar name exists, assume it's extracted
            potential_dir = os.path.join(NUSCENES_DIR, filename.split('.')[0])
            extract_archive(archive_path, NUSCENES_DIR, filename, use_pigz, logger)
    
    logger.info("✅ All nuScenes archives extracted.")

if __name__ == "__main__":
    main()