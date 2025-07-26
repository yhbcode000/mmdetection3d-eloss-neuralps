# kitti_download.py
import os
from tools.logging_setup import setup_logger
from tools.utils import download_file

DATA_ROOT = "data"
KITTI_DIR = os.path.join(DATA_ROOT, "kitti")
DOWNLOAD_DIR = os.path.join(KITTI_DIR, "downloads")
IMAGESETS_DIR = os.path.join(KITTI_DIR, "ImageSets")

KITTI_URLS = {
    "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
    "image_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "label_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
    "velodyne": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
}

IMAGESET_URLS = {
    "train.txt": "https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt",
    "val.txt": "https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt",
    "test.txt": "https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt",
    "trainval.txt": "https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt",
}

def main():
    logger = setup_logger()
    logger.info("🚙 Starting KITTI Dataset Download...")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(IMAGESETS_DIR, exist_ok=True)

    # Download ImageSets
    logger.info("📋 Downloading ImageSet files...")
    for filename, url in IMAGESET_URLS.items():
        output_path = os.path.join(IMAGESETS_DIR, filename)
        if os.path.exists(output_path):
            logger.info(f"✅ {filename} already exists, skipping.")
        else:
            download_file(url, output_path, f"KITTI {filename}", logger)

    # Download main data files
    logger.info("🎬 Downloading main data files...")
    for key, url in KITTI_URLS.items():
        filename = f"data_object_{key}.zip"
        output_path = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.exists(output_path):
            logger.info(f"✅ {filename} already exists, skipping.")
        else:
            download_file(url, output_path, f"KITTI {key}", logger)
            
    logger.info("✅ KITTI download process finished.")

if __name__ == "__main__":
    main()