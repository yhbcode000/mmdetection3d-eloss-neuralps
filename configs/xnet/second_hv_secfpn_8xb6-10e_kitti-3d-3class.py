_base_ = [
    # './models/second_hv_secfpn_kitti.py',
    './models/second_hv_secfpn_kitti_eloss.py',
    # './datasets/kitti-3d-3class.py',
    './datasets/kitti-3d-3class-noise.py',
    './schedules/cyclic-10e.py', 
    '../_base_/default_runtime.py'
]

load_from = "/workspace/mmdetection3d-eloss-neuralps2026/checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth"