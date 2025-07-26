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
    
    # Run data preprocessing if mmdet3d is available
    if check_mmdet3d_available(logger):
        logger.info("🔧 Running nuScenes data preprocessing...")
        try:
            # Assuming create_data.py is in a 'tools' subdir relative to project root
            # You may need to adjust the path to create_data.py
            create_script_path = "tools.create_data"
            subprocess.run(
                f"uv run python -m {create_script_path} nuscenes --root-path {NUSCENES_DIR} --out-dir {NUSCENES_DIR} --extra-tag nuscenes",
                shell=True, check=True
            )
            logger.info("✅ nuScenes preprocessing complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ nuScenes preprocessing failed: {e}")

    logger.info("✅ nuScenes setup complete.")

if __name__ == "__main__":
    main()