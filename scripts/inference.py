import os
import json
import mmcv
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mmengine import load
from mmengine.structures import InstanceData
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.apis import LidarDet3DInferencer

# --- 1. INITIALIZE MODEL AND DEFINE CLASS COLORS ---
config_path = 'configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
checkpoint_path = '/workspace/mmdetection3d/checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth'

print("Initializing model...")
inferencer = LidarDet3DInferencer(model=config_path, weights=checkpoint_path, device='cuda:0')

# Get the official class names from the model's metadata
class_names = inferencer.model.dataset_meta.get('classes', [])
print(f"Model class order: {class_names}")

# BGR Integer map for OpenCV/mmcv drawing
COLOR_MAP_BGR = {
    'car': (80, 127, 255),                 # Coral
    'truck': (255, 255, 0),                # Cyan
    'construction_vehicle': (219, 112, 147),# Medium Purple
    'bus': (255, 191, 0),                  # Deep Sky Blue
    'trailer': (147, 20, 255),             # Deep Pink
    'barrier': (0, 215, 255),              # Gold
    'motorcycle': (154, 250, 0),           # Medium Spring Green
    'bicycle': (139, 61, 72),              # Dark Slate Blue
    'pedestrian': (0, 69, 255),            # Orange Red
    'traffic_cone': (0, 128, 0),           # Green
}

# RGB Float map for Matplotlib legend
COLOR_MAP_RGB = {name: tuple(c/255. for c in bgr[::-1]) for name, bgr in COLOR_MAP_BGR.items()}


# --- 2. DEFINE DATA PATHS ---
data_root = '/workspace/mmdetection3d/demo/data/nuscenes/'
data_name = 'n015-2018-07-24-11-22-45+0800'
out_dir = './outputs_nuscenes'
camera_type = 'CAM_FRONT'

image_path = '/workspace/mmdetection3d/demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg'
lidar_path_full = '/workspace/mmdetection3d/demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin'
info_path = os.path.join(data_root, f'{data_name}.pkl')


# --- 3. LOAD INFO FOR CALIBRATION ---
print(f"Loading calibration info from: {info_path}")
info_file = load(info_path)
sample_info = info_file['data_list'][0]


# --- 4. RUN INFERENCE ---
print(f"Running inference on: {lidar_path_full}")
inferencer(dict(points=lidar_path_full), show=False, out_dir=out_dir)


# --- 5. LOAD PREDICTION AND CALIBRATION DATA ---
lidar_filename_stem = os.path.splitext(os.path.basename(lidar_path_full))[0]
pred_json_path = os.path.join(out_dir, 'preds', f'{lidar_filename_stem}.json')

print(f"\nLoading prediction data from: {pred_json_path}")
with open(pred_json_path, 'r') as f:
    pred_data = json.load(f)

print(f"Loading calibration data for {camera_type}...")
cam_info = sample_info['images'][camera_type]
cam2img = np.array(cam_info['cam2img'], dtype=np.float32)
lidar2cam = np.array(cam_info['lidar2cam'], dtype=np.float32)


# --- 6. PROCESS CALIBRATION MATRICES ---
cam2img_4x4 = np.eye(4, dtype=np.float32)
if cam2img.shape == (3, 3):
    cam2img_4x4[:3, :3] = cam2img
elif cam2img.shape == (3, 4):
    cam2img_4x4[:3, :4] = cam2img
elif cam2img.shape == (4, 4):
    cam2img_4x4 = cam2img
else:
    raise ValueError(f"Unsupported shape for cam2img: {cam2img.shape}")

lidar2img = cam2img_4x4 @ lidar2cam
input_meta = {'cam2img': cam2img, 'lidar2cam': lidar2cam, 'lidar2img': lidar2img}


# --- 7. PREPARE AND FILTER PREDICTIONS ---
img = mmcv.imread(image_path)
img = mmcv.imconvert(img, 'bgr', 'rgb')

pred_instances = InstanceData()
bboxes_3d = np.array(pred_data['bboxes_3d'], dtype=np.float32)
scores_3d = np.array(pred_data['scores_3d'], dtype=np.float32)
labels_3d = np.array(pred_data['labels_3d'], dtype=np.int64)

pred_instances.bboxes_3d = LiDARInstance3DBoxes(bboxes_3d, box_dim=9)
pred_instances.scores_3d = torch.from_numpy(scores_3d)
pred_instances.labels_3d = torch.from_numpy(labels_3d)

score_threshold = 0.3
mask = pred_instances.scores_3d >= score_threshold
filtered_instances = pred_instances[mask]
print(f"Filtered {len(pred_instances)} predictions down to {len(filtered_instances)} with score > {score_threshold}")


# --- 8. VISUALIZE WITH CLASS-SPECIFIC COLORS ---
print("\nInitializing visualizer and drawing projected 3D boxes...")
visualizer = Det3DLocalVisualizer()
visualizer.set_image(img)

box_colors = []
for label_index in filtered_instances.labels_3d:
    class_name = class_names[label_index]
    color = COLOR_MAP_BGR.get(class_name, (0, 0, 0))
    box_colors.append(color)

visualizer.draw_proj_bboxes_3d(
    filtered_instances.bboxes_3d,
    input_meta=input_meta,
    edge_colors=box_colors,
    face_colors=box_colors,
    alpha=0.4
)
drawn_img = visualizer.get_image()


# --- ✨ 9. SAVE VISUALIZATION TO FILE ---
# Create legend handles for the plot
legend_patches = []
for name in class_names:
    if name in COLOR_MAP_RGB:
        patch = mpatches.Patch(color=COLOR_MAP_RGB[name], label=name)
        legend_patches.append(patch)

# Set up the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(24, 8))

axes[0].imshow(img)
axes[0].set_title(f"Original nuScenes Image ({camera_type})")
axes[0].axis('off')

axes[1].imshow(drawn_img)
axes[1].set_title(f"Plot with Filtered 3D Bounding Boxes (Score > {score_threshold})")
axes[1].axis('off')
axes[1].legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.9, 1])

# Define the output path and save the figure
vis_dir = os.path.join(out_dir, 'vis')
os.makedirs(vis_dir, exist_ok=True)
output_path = os.path.join(vis_dir, f'{lidar_filename_stem}_visualization.png')
print(f"\nSaving visualization to: {output_path}")
plt.savefig(output_path, bbox_inches='tight', dpi=150)
plt.close(fig)  # Close the figure to free up memory