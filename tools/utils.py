# tools/utils.py
import os
import shutil
import subprocess
import logging
import zipfile
import tarfile

# --- System & Dependency Checks ---

def check_pigz_available(logger):
    """Checks if pigz is available for parallel decompression."""
    try:
        result = subprocess.run(['pigz', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("✅ pigz is available, will use parallel extraction.")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    logger.info("⚠️ pigz is not available, using standard extraction.")
    logger.info("💡 Tip: Install pigz for faster extraction (e.g., 'sudo apt-get install pigz').")
    return False

def check_mmdet3d_available(logger):
    """Checks if MMDetection3D is installed."""
    try:
        import mmdet3d
        logger.info("✅ MMDetection3D is installed.")
        return True
    except ImportError:
        logger.warning("⚠️ MMDetection3D not found, skipping data preprocessing.")
        logger.info("💡 To run preprocessing, please install it via 'pip install mmdet3d'.")
        return False

# --- File Operations ---

def download_file(url, output_path, description, logger):
    """Downloads a file using aria2c for multi-connection download."""
    logger.info(f"⬇️  Starting download for {description}...")
    dir_name = os.path.dirname(output_path)
    file_name = os.path.basename(output_path)
    
    cmd = f'aria2c -x 16 -s 16 -c -d "{dir_name}" -o "{file_name}" "{url}"'
    subprocess.Popen(cmd, shell=True)
    

def extract_archive(archive_path, extract_to, description, use_pigz, logger):
    """Extracts an archive, using pigz if available."""
    if not os.path.exists(archive_path):
        logger.error(f"❌ Cannot extract, archive not found: {archive_path}")
        return False

    logger.info(f"📂 Extracting {description} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        if archive_path.endswith(('.tar.gz', '.tgz')):
            if use_pigz:
                cmd = f'tar --no-same-owner --use-compress-program=pigz -xf "{archive_path}" -C "{extract_to}"'
                subprocess.Popen(cmd, shell=True)
            else:
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(path=extract_to)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar:
                tar.extractall(path=extract_to)
        else:
            logger.error(f"❌ Unsupported archive format: {archive_path}")
            return False
            
        logger.info(f"✅ {description} extracted successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ An error occurred during extraction of {description}: {e}")
        return False

def count_files_in_directory(directory, logger):
    """Recursively counts files in a directory."""
    try:
        count = 0
        for _, _, files in os.walk(directory):
            count += len(files)
        return count
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not access directory {directory}: {e}")
        return 0

# --- Dataset Specific Helpers ---

def organize_kitti_structure(kitti_dir, logger):
    """Ensures the KITTI directory structure is correct after extraction."""
    logger.info("📁 Organizing KITTI directory structure...")
    
    training_dir = os.path.join(kitti_dir, "training")
    testing_dir = os.path.join(kitti_dir, "testing")
    
    # Ensure base directories exist
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(testing_dir, exist_ok=True)

    # Move content from nested training/testing folders if they exist
    for split in ["training", "testing"]:
        nested_dir = os.path.join(kitti_dir, split, split)
        if os.path.exists(nested_dir):
            logger.info(f"Moving files from nested '{nested_dir}' to '{os.path.join(kitti_dir, split)}'")
            for item in os.listdir(nested_dir):
                shutil.move(os.path.join(nested_dir, item), os.path.join(kitti_dir, split))
            os.rmdir(nested_dir)

    # Copy calib data to testing directory if it exists in training
    training_calib = os.path.join(training_dir, "calib")
    testing_calib = os.path.join(testing_dir, "calib")
    if os.path.exists(training_calib) and not os.path.exists(testing_calib):
        logger.info(" Copying 'calib' directory to 'testing' folder.")
        shutil.copytree(training_calib, testing_calib)