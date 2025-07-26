# kitti_extract.py
import os
import subprocess
from tools.logging_setup import setup_logger
from tools.utils import check_pigz_available, extract_archive, organize_kitti_structure, check_mmdet3d_available

DATA_ROOT = "data"
KITTI_DIR = os.path.join(DATA_ROOT, "kitti")
DOWNLOAD_DIR = os.path.join(KITTI_DIR, "downloads")

def main():
    logger = setup_logger()
    logger.info("🚙 Starting KITTI Dataset Extraction and Setup...")

    if not os.path.exists(DOWNLOAD_DIR) or not os.listdir(DOWNLOAD_DIR):
        logger.error("❌ Download directory is empty. Please run 'kitti_download.py' first.")
        return

    use_pigz = check_pigz_available(logger) # pigz is for .gz, but we keep the check for consistency

    # Extract all .zip files
    for filename in os.listdir(DOWNLOAD_DIR):
        if filename.endswith('.zip'):
            archive_path = os.path.join(DOWNLOAD_DIR, filename)
            # Check if corresponding data folder exists to prevent re-extraction
            key = filename.replace('data_object_', '').replace('.zip', '')
            potential_dir = os.path.join(KITTI_DIR, "training", key)
            if key == 'calib': # calib is a special case
                 potential_dir = os.path.join(KITTI_DIR, "training", "calib")
            extract_archive(archive_path, KITTI_DIR, filename, use_pigz, logger)
    
    logger.info("✅ All KITTI archives extracted.")

if __name__ == "__main__":
    main()