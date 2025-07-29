# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_multi-SECOND-3class.py',
    '../_base_/schedules/cyclic_60e.py', 
    '../_base_/default_runtime.py'
    ]

# concate + structure loss
# load_from = "work_dirs/xnet_exp00-kitti_3d_3class-cyclic_40e/epoch_4.pth"

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4
    )