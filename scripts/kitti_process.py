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

    # Organize directory structure
    organize_kitti_structure(KITTI_DIR, logger)
    
    # Run data preprocessing if mmdet3d is available
    if check_mmdet3d_available(logger):
        logger.info("🔧 Running KITTI data preprocessing...")
        try:
            # You may need to adjust the path to create_data.py
            create_script_path = "tools.create_data"
            subprocess.run(
                f"uv run python -m {create_script_path} kitti --root-path {KITTI_DIR} --out-dir {KITTI_DIR} --extra-tag kitti",
                shell=True, check=True
            )
            logger.info("✅ KITTI preprocessing complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ KITTI preprocessing failed: {e}")

    logger.info("✅ KITTI setup complete.")

if __name__ == "__main__":
    main()