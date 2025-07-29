# Add these imports at the top of your transforms file if they don't exist
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import BasePoints
from typing import List


@TRANSFORMS.register_module()
class AdditiveGaussianNoise(BaseTransform):
    """Applies additive Gaussian noise to each point's coordinate.

    This transform adds a random noise vector, sampled from a Gaussian
    distribution, to the XYZ coordinates of every point in the point cloud.
    It is functionally similar to point jittering.

    Required Keys:
    - points

    Modified Keys:
    - points

    Args:
        std (List[float]): The standard deviation of the Gaussian noise for
            the X, Y, and Z axes. Defaults to [0.01, 0.01, 0.01].
        clip_range (List[float], optional): The range `[min, max]` to clip the
            generated noise. If None, no clipping is performed.
            Defaults to [-0.05, 0.05].
    """

    def __init__(self,
                 std: List[float] = [0.01, 0.01, 0.01],
                 clip_range: List[float] = [-0.05, 0.05]) -> None:
        self.std = std
        self.clip_range = clip_range

    def transform(self, input_dict: dict) -> dict:
        """Applies the transformation.

        Args:
            input_dict (dict): A dictionary containing the points.

        Returns:
            dict: The dictionary with noisy points.
        """
        points = input_dict['points']
        
        # Generate Gaussian noise
        noise_std = np.array(self.std, dtype=np.float32)
        noise = np.random.randn(points.shape[0], 3) * noise_std

        # Clip noise if a range is specified
        if self.clip_range is not None:
            noise = np.clip(noise, self.clip_range[0], self.clip_range[1])

        # Add noise to point coordinates
        points.translate(noise)
        input_dict['points'] = points
        
        return input_dict

    def __repr__(self) -> str:
        """String representation of the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(std={self.std}, clip_range={self.clip_range})'
        return repr_str


@TRANSFORMS.register_module()
class SaltAndPepperNoise(BaseTransform):
    """Applies salt and pepper noise to a point cloud.

    "Salt" noise adds a number of new, randomly placed points.
    "Pepper" noise removes a random fraction of existing points.

    Required Keys:
    - points

    Modified Keys:
    - points

    Args:
        salt_ratio (float): The ratio of new points to add, relative to the
            original number of points. Defaults to 0.05 (adds 5% new points).
        pepper_ratio (float): The ratio of existing points to remove.
            Defaults to 0.05 (removes 5% of original points).
    """

    def __init__(self,
                 salt_ratio: float = 0.05,
                 pepper_ratio: float = 0.05) -> None:
        assert 0 <= salt_ratio, 'salt_ratio must be non-negative'
        assert 0 <= pepper_ratio < 1, 'pepper_ratio must be between 0 and 1'
        self.salt_ratio = salt_ratio
        self.pepper_ratio = pepper_ratio

    def transform(self, input_dict: dict) -> dict:
        """Applies the transformation.

        Args:
            input_dict (dict): A dictionary containing the points.

        Returns:
            dict: The dictionary with salt-and-pepper noisy points.
        """
        points = input_dict['points']
        num_points = len(points)
        
        # --- Pepper Noise (Remove Points) ---
        if self.pepper_ratio > 0:
            num_pepper = int(num_points * self.pepper_ratio)
            if num_pepper > 0:
                # Choose random indices to drop without replacement
                drop_indices = np.random.choice(
                    np.arange(num_points), num_pepper, replace=False)
                
                # Create a mask to keep the remaining points
                keep_mask = np.ones(num_points, dtype=bool)
                keep_mask[drop_indices] = False
                points = points[keep_mask]

        # --- Salt Noise (Add Points) ---
        if self.salt_ratio > 0:
            num_salt = int(num_points * self.salt_ratio)
            if num_salt > 0:
                # Get the bounding box of the original point cloud
                # to generate new points within the same space.
                min_coords = points.coord.min(axis=0).numpy()
                max_coords = points.coord.max(axis=0).numpy()
                
                # Generate new random coordinates
                salt_coords = np.random.uniform(
                    low=min_coords, high=max_coords, size=(num_salt, 3))
                
                # Create new points with other attributes (e.g., intensity) as zero
                point_dim = points.tensor.shape[1]
                salt_tensor = np.zeros((num_salt, point_dim), dtype=np.float32)
                salt_tensor[:, :3] = salt_coords

                # Create a new BasePoints object and concatenate
                salt_points = points.new_point(salt_tensor)
                points = points.cat([points, salt_points])

        input_dict['points'] = points
        return input_dict

    def __repr__(self) -> str:
        """String representation of the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(salt_ratio={self.salt_ratio}, '
        repr_str += f'pepper_ratio={self.pepper_ratio})'
        return repr_str