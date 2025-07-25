import os
import sys
import logging
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import zipfile
import tarfile

# 日志配置
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.flush = sys.stdout.flush
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

class DatasetDownloader:
    def __init__(self, dataset_type="both"):
        self.dataset_type = dataset_type
        self.data_root = "data"
        self.nuscenes_dir = os.path.join(self.data_root, "nuscenes")
        self.kitti_dir = os.path.join(self.data_root, "kitti")
        
        # Create directories
        os.makedirs(self.data_root, exist_ok=True)
        os.makedirs(self.nuscenes_dir, exist_ok=True)
        os.makedirs(self.kitti_dir, exist_ok=True)
        
        logger.info(f"📁 当前工作目录：{os.getcwd()}")
        logger.info(f"📂 数据集根目录：{self.data_root}")

    def download_file(self, url, output_path, description=""):
        """下载文件使用 aria2c"""
        logger.info(f"⬇️ 开始下载 {description}...")
        cmd = f'aria2c -x 16 -s 16 -c -d "{os.path.dirname(output_path)}" -o "{os.path.basename(output_path)}" "{url}"'
        result = os.system(cmd)
        
        if result == 0 and os.path.exists(output_path):
            logger.info(f"✅ {description} 下载完成：{os.path.basename(output_path)}")
            return True
        else:
            logger.error(f"❌ {description} 下载失败")
            return False

    def extract_archive(self, archive_path, extract_to, description=""):
        """解压文件"""
        logger.info(f"📂 正在解压 {description}...")
        
        if archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar:
                tar.extractall(extract_to)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_to)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            logger.error(f"❌ 不支持的文件格式：{archive_path}")
            return False
            
        logger.info(f"✅ {description} 解压完成")
        return True

    def get_nuscenes_download_url(self):
        """获取用户输入的 nuScenes 下载 URL"""
        logger.info("🔗 nuScenes 数据集下载需要用户提供访问 URL")
        logger.info("📝 请访问 https://www.nuscenes.org/nuscenes 注册账号")
        logger.info("🌐 基础 CDN URL: https://d36yt3mvayqw5m.cloudfront.net/public/v1.0")
        
        print("\n" + "="*60)
        print("请输入您的 nuScenes 下载访问 URL:")
        print("格式示例: https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/your-access-token")
        print("或者直接输入访问令牌部分")
        print("="*60)
        
        user_input = input("请输入 URL 或访问令牌: ").strip()
        
        if not user_input:
            logger.warning("⚠️ 未输入 URL，将跳过自动下载")
            return None
        
        # 检查是否是完整 URL
        if user_input.startswith("http"):
            base_url = user_input.rstrip('/')
        else:
            # 假设是访问令牌，拼接完整 URL
            base_url = f"https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/{user_input}"
        
        logger.info(f"✅ 设置下载基础 URL: {base_url}")
        return base_url

    def setup_nuscenes(self):
        """设置 nuScenes 数据集"""
        logger.info("🚗 开始设置 nuScenes 数据集...")
        
        # 获取用户输入的下载 URL
        base_url = self.get_nuscenes_download_url()
        
        if base_url is None:
            logger.warning("⚠️ 未提供下载 URL，将检查本地文件")
            self._process_existing_nuscenes_files()
            return
        
        # nuScenes 文件列表 (更新为正确的文件名)
        nuscenes_files = {
            "trainval_meta": "v1.0-trainval_meta.tgz",
            "trainval01": "v1.0-trainval01_blobs.tgz", 
            "trainval02": "v1.0-trainval02_blobs.tgz",
            "trainval03": "v1.0-trainval03_blobs.tgz",
            "trainval04": "v1.0-trainval04_blobs.tgz",
            "trainval05": "v1.0-trainval05_blobs.tgz",
            "trainval06": "v1.0-trainval06_blobs.tgz",
            "trainval07": "v1.0-trainval07_blobs.tgz",
            "trainval08": "v1.0-trainval08_blobs.tgz",
            "trainval09": "v1.0-trainval09_blobs.tgz",
            "trainval10": "v1.0-trainval10_blobs.tgz",
            "test_blobs": "v1.0-test_blobs.tgz",
            "test_meta": "v1.0-test_meta.tgz",
            "mini": "v1.0-mini.tgz"
        }
        
        # 创建下载目录
        download_dir = os.path.join(self.nuscenes_dir, "downloads")
        os.makedirs(download_dir, exist_ok=True)
        
        # 下载文件
        for key, filename in nuscenes_files.items():
            file_url = f"{base_url}/{filename}"
            output_path = os.path.join(download_dir, filename)
            
            if not os.path.exists(output_path):
                logger.info(f"⬇️ 下载 {key}...")
                if not self.download_file(file_url, output_path, f"nuScenes {key}"):
                    logger.warning(f"⚠️ {key} 下载失败，跳过")
                    continue
            else:
                logger.info(f"✅ {key} 已存在，跳过下载")
        
        # 处理下载的文件
        self._process_existing_nuscenes_files()

    def _process_existing_nuscenes_files(self):
        """处理已存在的 nuScenes 文件"""
        download_dir = os.path.join(self.nuscenes_dir, "downloads")
        os.makedirs(download_dir, exist_ok=True)
        
        downloaded_files = []
        for file in os.listdir(download_dir):
            if file.endswith('.tgz'):
                downloaded_files.append(os.path.join(download_dir, file))
        
        if downloaded_files:
            logger.info(f"🔍 发现 {len(downloaded_files)} 个已下载的文件，开始解压...")
            for file_path in downloaded_files:
                self.extract_archive(file_path, self.nuscenes_dir, os.path.basename(file_path))
        else:
            logger.warning("⚠️ 未找到 nuScenes 数据文件")
            logger.info("💡 请手动下载文件到 data/nuscenes/downloads/ 目录")
        
        # 创建目录结构
        required_dirs = ['maps', 'samples', 'sweeps', 'v1.0-test', 'v1.0-trainval']
        for dir_name in required_dirs:
            os.makedirs(os.path.join(self.nuscenes_dir, dir_name), exist_ok=True)
        
        # 运行数据处理脚本
        if self._check_mmdet3d_available():
            logger.info("🔧 运行 nuScenes 数据预处理...")
            cmd = f"python tools/create_data.py nuscenes --root-path {self.nuscenes_dir} --out-dir {self.nuscenes_dir} --extra-tag nuscenes"
            os.system(cmd)
        
        logger.info("✅ nuScenes 数据集设置完成")

    def setup_kitti(self):
        """设置 KITTI 数据集"""
        logger.info("🚙 开始设置 KITTI 数据集...")
        
        # KITTI 下载链接
        kitti_urls = {
            "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
            "image_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
            "label_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
            "velodyne": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
        }
        
        # 下载数据分割文件
        imagesets_dir = os.path.join(self.kitti_dir, "ImageSets")
        os.makedirs(imagesets_dir, exist_ok=True)
        
        imageset_urls = {
            "train.txt": "https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt",
            "val.txt": "https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt",
            "test.txt": "https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt",
            "trainval.txt": "https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt",
        }
        
        # 下载数据分割文件
        for filename, url in imageset_urls.items():
            output_path = os.path.join(imagesets_dir, filename)
            if not os.path.exists(output_path):
                self.download_file(url, output_path, f"KITTI {filename}")
        
        # 下载主要数据文件
        downloads_dir = os.path.join(self.kitti_dir, "downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        
        for key, url in kitti_urls.items():
            filename = f"data_object_{key}.zip"
            output_path = os.path.join(downloads_dir, filename)
            
            if not os.path.exists(output_path):
                if self.download_file(url, output_path, f"KITTI {key}"):
                    # 解压到对应目录
                    if key == "calib":
                        extract_to = os.path.join(self.kitti_dir, "training")
                        os.makedirs(extract_to, exist_ok=True)
                        self.extract_archive(output_path, extract_to, f"KITTI {key}")
                        
                        # 创建 testing 目录并复制 calib
                        testing_dir = os.path.join(self.kitti_dir, "testing")
                        os.makedirs(testing_dir, exist_ok=True)
                        testing_calib = os.path.join(testing_dir, "calib")
                        training_calib = os.path.join(extract_to, "training", "calib")
                        if os.path.exists(training_calib) and not os.path.exists(testing_calib):
                            shutil.copytree(training_calib, testing_calib)
                    else:
                        # 其他文件直接解压到 kitti 根目录
                        self.extract_archive(output_path, self.kitti_dir, f"KITTI {key}")
        
        # 整理目录结构
        self._organize_kitti_structure()
        
        # 运行数据处理脚本
        if self._check_mmdet3d_available():
            logger.info("🔧 运行 KITTI 数据预处理...")
            cmd = f"python tools/create_data.py kitti --root-path {self.kitti_dir} --out-dir {self.kitti_dir} --extra-tag kitti"
            os.system(cmd)
        
        logger.info("✅ KITTI 数据集设置完成")

    def _organize_kitti_structure(self):
        """整理 KITTI 目录结构"""
        logger.info("📁 整理 KITTI 目录结构...")
        
        # 确保正确的目录结构
        training_dir = os.path.join(self.kitti_dir, "training")
        testing_dir = os.path.join(self.kitti_dir, "testing")
        
        for split_dir in [training_dir, testing_dir]:
            for subdir in ["calib", "image_2", "velodyne"]:
                os.makedirs(os.path.join(split_dir, subdir), exist_ok=True)
        
        # 为 training 添加 label_2
        os.makedirs(os.path.join(training_dir, "label_2"), exist_ok=True)
        
        # 检查是否需要移动文件
        source_training = os.path.join(self.kitti_dir, "training", "training")
        if os.path.exists(source_training):
            # 移动文件到正确位置
            for item in os.listdir(source_training):
                src = os.path.join(source_training, item)
                dst = os.path.join(training_dir, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            # 删除空的嵌套目录
            if not os.listdir(source_training):
                os.rmdir(source_training)

    def _check_mmdet3d_available(self):
        """检查 MMDetection3D 是否可用"""
        try:
            import mmdet3d
            return True
        except ImportError:
            logger.warning("⚠️ MMDetection3D 未安装，跳过数据预处理步骤")
            logger.info("💡 请安装 MMDetection3D 后手动运行数据预处理：")
            logger.info("   pip install mmdet3d")
            return False

    def verify_datasets(self):
        """验证数据集完整性"""
        logger.info("🔍 验证数据集完整性...")
        
        if self.dataset_type in ["nuscenes", "both"]:
            self._verify_nuscenes()
        
        if self.dataset_type in ["kitti", "both"]:
            self._verify_kitti()

    def _verify_nuscenes(self):
        """验证 nuScenes 数据集"""
        required_dirs = ['maps', 'samples', 'sweeps', 'v1.0-trainval']
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = os.path.join(self.nuscenes_dir, dir_name)
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            logger.warning(f"⚠️ nuScenes 缺少目录: {missing_dirs}")
        else:
            logger.info("✅ nuScenes 目录结构完整")
        
        # 检查 pkl 文件
        pkl_files = ['nuscenes_infos_train.pkl', 'nuscenes_infos_val.pkl']
        for pkl_file in pkl_files:
            if os.path.exists(os.path.join(self.nuscenes_dir, pkl_file)):
                logger.info(f"✅ 找到 {pkl_file}")
            else:
                logger.warning(f"⚠️ 缺少 {pkl_file}")

    def _verify_kitti(self):
        """验证 KITTI 数据集"""
        required_structure = {
            'ImageSets': ['train.txt', 'val.txt', 'test.txt', 'trainval.txt'],
            'training': ['calib', 'image_2', 'label_2', 'velodyne'],
            'testing': ['calib', 'image_2', 'velodyne']
        }
        
        for parent_dir, subdirs in required_structure.items():
            parent_path = os.path.join(self.kitti_dir, parent_dir)
            if not os.path.exists(parent_path):
                logger.warning(f"⚠️ KITTI 缺少目录: {parent_dir}")
                continue
                
            for subdir in subdirs:
                subdir_path = os.path.join(parent_path, subdir)
                if os.path.exists(subdir_path):
                    if os.path.isdir(subdir_path):
                        file_count = len(os.listdir(subdir_path))
                        logger.info(f"✅ {parent_dir}/{subdir}: {file_count} 个文件")
                    else:
                        logger.info(f"✅ 找到 {parent_dir}/{subdir}")
                else:
                    logger.warning(f"⚠️ 缺少 {parent_dir}/{subdir}")
        
        # 检查 pkl 文件
        pkl_files = ['kitti_infos_train.pkl', 'kitti_infos_val.pkl', 'kitti_infos_test.pkl']
        for pkl_file in pkl_files:
            if os.path.exists(os.path.join(self.kitti_dir, pkl_file)):
                logger.info(f"✅ 找到 {pkl_file}")
            else:
                logger.warning(f"⚠️ 缺少 {pkl_file}")

    def run(self):
        """运行下载流程"""
        logger.info(f"🚀 开始下载 {self.dataset_type} 数据集...")
        
        if self.dataset_type in ["nuscenes", "both"]:
            self.setup_nuscenes()
        
        if self.dataset_type in ["kitti", "both"]:
            self.setup_kitti()
        
        self.verify_datasets()
        
        logger.info("🎉 数据集准备完成！")
        logger.info("📖 使用说明：")
        logger.info("   - nuScenes: 需要提供访问 URL 或手动下载文件")
        logger.info("   - KITTI: 自动下载完成")
        logger.info("   - 如安装了 MMDetection3D，数据预处理会自动运行")


def main():
    parser = argparse.ArgumentParser(description="下载并设置 nuScenes 和 KITTI 数据集")
    parser.add_argument(
        "--dataset", 
        choices=["nuscenes", "kitti", "both"], 
        default="both",
        help="选择要下载的数据集 (默认: both)"
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.dataset)
    downloader.run()


if __name__ == "__main__":
    main()
