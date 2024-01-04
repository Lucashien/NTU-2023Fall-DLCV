import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
import glob
from PIL import Image
from torchvision import transforms as T
import random
from kornia import create_meshgrid
from scipy.spatial import transform


def calculate_near_and_far(
    rays_o, rays_d, bbox_min=[-1.0, -1.0, -1.0], bbox_max=[1.0, 1.0, 1.0]
):
    """
    rays_o, (len(self.split_ids)*h*w, 3)
    rays_d, (len(self.split_ids)*h*w, 3)
    bbox_min=[-1,-1,-1],
    bbox_max=[1,1,1]
    """
    # map all shape to same (len(self.split_ids)*h*w, 3, 2)
    corners = torch.stack((torch.tensor(bbox_min), torch.tensor(bbox_max)), dim=-1)
    corners = corners.unsqueeze(0).repeat(
        rays_o.shape[0], 1, 1
    )  # (len(self.split_ids)*h*w, 3, 2)
    corners -= torch.unsqueeze(rays_o, -1).repeat(1, 1, 2)
    intersections = corners / (torch.unsqueeze(rays_d, -1).repeat(1, 1, 2))

    min_intersections = torch.amax(
        torch.amin(intersections, dim=-1), dim=-1, keepdim=True
    )
    max_intersections = torch.amin(
        torch.amax(intersections, dim=-1), dim=-1, keepdim=True
    )
    epsilon = 1e-10 * torch.ones_like(min_intersections)
    near = torch.maximum(epsilon, min_intersections)
    # tmp = near
    near = torch.where((near > max_intersections), epsilon, near)
    far = torch.where(near < max_intersections, max_intersections, near + epsilon)

    return near, far


def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = torch.stack(
        [(i - W / 2) / focal, -(j - H / 2) / focal, -torch.ones_like(i)], -1
    )  # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def blender_quat2rot(quaternion):
    """Convert quaternion to rotation matrix.
    Equivalent to, but support batched case:
    ```python
    rot3x3 = mathutils.Quaternion(quaternion).to_matrix()
    ```
    Args:
    quaternion:
    Returns:
    rotation matrix
    """

    # Note: Blender first cast to double values for numerical precision while
    # we're using float32.
    q = np.sqrt(2) * quaternion

    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    qda = q0 * q1
    qdb = q0 * q2
    qdc = q0 * q3
    qaa = q1 * q1
    qab = q1 * q2
    qac = q1 * q3
    qbb = q2 * q2
    qbc = q2 * q3
    qcc = q3 * q3

    # Note: idx are inverted as blender and numpy convensions do not
    # match (x, y) -> (y, x)
    rotation = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float32)
    rotation[..., 0, 0] = 1.0 - qbb - qcc
    rotation[..., 1, 0] = qdc + qab
    rotation[..., 2, 0] = -qdb + qac

    rotation[..., 0, 1] = -qdc + qab
    rotation[..., 1, 1] = 1.0 - qaa - qcc
    rotation[..., 2, 1] = qda + qbc

    rotation[..., 0, 2] = qdb + qac
    rotation[..., 1, 2] = -qda + qbc
    rotation[..., 2, 2] = 1.0 - qaa - qbb
    return rotation


def make_transform_matrix(
    positions,
    rotations,
):
    """Create the 4x4 transformation matrix.
    Note: This function uses numpy.
    Args:
    positions: Translation applied after the rotation.
        Last column of the transformation matrix
    rotations: Rotation. Top-left 3x3 matrix of the transformation matrix.
    Returns:
    transformation_matrix:
    """
    # Create the 4x4 transformation matrix
    rot_pos = np.broadcast_to(np.eye(4), (*positions.shape[:-1], 4, 4)).copy()
    rot_pos[..., :3, :3] = rotations
    rot_pos[..., :3, 3] = positions
    return rot_pos


def from_position_and_quaternion(positions, quaternions, use_unreal_axes):
    if use_unreal_axes:
        rotations = transform.Rotation.from_quat(quaternions).as_matrix()
    else:
        # Rotation matrix that rotates from world to object coordinates.
        # Warning: Rotations should be given in blender convensions as
        # scipy.transform uses different convensions.
        rotations = blender_quat2rot(quaternions)
    px2world_transform = make_transform_matrix(positions=positions, rotations=rotations)
    return px2world_transform


def scale_rays(all_rays_o, all_rays_d, scene_boundaries, img_wh):
    """Rescale scene boundaries.
    rays_o: (len(image_paths)*h*w, 3)
    rays_d: (len(image_paths)*h*w, 3)
    scene_boundaries: np.array(2 ,3), [min, max]
    img_wh: (2)
    """
    # Rescale (x, y, z) from [min, max] -> [-1, 1]
    all_rays_o = all_rays_o.reshape(
        -1, img_wh[0], img_wh[1], 3
    )  # (len(image_paths), h, w, 3))
    all_rays_d = all_rays_d.reshape(-1, img_wh[0], img_wh[1], 3)

    old_min = torch.from_numpy(scene_boundaries[0])
    old_max = torch.from_numpy(scene_boundaries[1])
    new_min = torch.tensor([-1, -1, -1])
    new_max = torch.tensor([1, 1, 1])
    # scale = max(scene_boundaries[1] - scene_boundaries[0])/2
    # all_rays_o = (all_rays_o - torch.mean(all_rays_o, dim=-1, keepdim=True)) / scale
    # This is from jax3d.interp, kind of weird but true
    all_rays_o = ((new_min - new_max) / (old_min - old_max)) * all_rays_o + (
        old_min * new_max - new_min * old_max
    ) / (old_min - old_max)

    # We also need to rescale the camera direction by bbox.size.
    # The direction can be though of a ray from a point in space (the camera
    # origin) to another point in space (say the red light on the lego
    # bulldozer). When we scale the scene in a certain way, this direction
    # also needs to be scaled in the same way.
    all_rays_d = all_rays_d * 2 / (scene_boundaries[1] - scene_boundaries[0])
    # (re)-normalize the rays
    all_rays_d = all_rays_d / torch.linalg.norm(all_rays_d, dim=-1, keepdims=True)
    return all_rays_o.reshape(-1, 3), all_rays_d.reshape(-1, 3)


# Nesf Klevr


class KlevrDataset(Dataset):
    def __init__(self, root_dir, split="train", get_rgb=True) -> None:
        # super().__init__()
        """
        split: train/val/test
        """
        self.root_dir = root_dir
        self.get_rgb = get_rgb
        self.split = split

        self.define_transforms()
        self.read_meta()

    def define_transforms(self):
        self.transform = T.ToTensor()

    def read_meta(self):
        with open(os.path.join(self.root_dir, "metadata.json"), "r") as f:
            self.meta = json.load(f)

        w, h = self.meta["metadata"]["width"], self.meta["metadata"]["width"]
        self.img_wh = (w, h)
        self.focal = (
            self.meta["camera"]["focal_length"]
            * w
            / self.meta["camera"]["sensor_width"]
        )
        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"split should be train/val/test, got {self.split}")
        self.split_ids = self.meta["split_ids"][self.split]

        self.scene_boundaries = np.array(
            [self.meta["scene_boundaries"]["min"], self.meta["scene_boundaries"]["max"]]
        )
        self.directions = get_ray_directions(h, w, self.focal)

        if self.split == "train":
            self.poses = []
            self.all_rays_o = []
            self.all_rays_d = []
            self.all_rays = []
            self.all_rgbs = []
            camera_positions = np.array(self.meta["camera"]["positions"])
            camera_quaternions = np.array(self.meta["camera"]["quaternions"])
            for image_id in self.split_ids:
                if self.get_rgb:
                    image_path = os.path.join(self.root_dir, f"{image_id:05d}.png")
                    img = Image.open(image_path)
                    img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
                    img = self.transform(img)  # (4, h, w)
                    img = img.view(4, -1).permute(1, 0)  # (h*w, 4)
                    # not sure, original jax implementation seems not using blend just cut it off, they also /255 to make it [0,1] which I didn't use
                    # img = img[:, :3]*img[:, -1:]+(1-img[:,-1:]) # blend A to RGB
                    img = img[:, :3]
                    self.all_rgbs += [img]

                pose = np.array(
                    from_position_and_quaternion(
                        camera_positions[image_id : image_id + 1, :],
                        camera_quaternions[image_id : image_id + 1, :],
                        False,
                    )
                )[0, :3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)
                rays_o, rays_d = get_rays(self.directions, c2w)
                self.all_rays_o += [rays_o]
                self.all_rays_d += [rays_d]

            self.all_rays_o = torch.cat(
                self.all_rays_o, 0
            )  # (len(self.split_ids)*h*w, 3)
            self.all_rays_d = torch.cat(
                self.all_rays_d, 0
            )  # (len(self.split_ids)*h*w, 3)
            self.all_rays_o, self.all_rays_d = scale_rays(
                self.all_rays_o, self.all_rays_d, self.scene_boundaries, self.img_wh
            )

            self.near, self.far = calculate_near_and_far(
                self.all_rays_o, self.all_rays_d
            )
            self.all_rays = torch.cat(
                [self.all_rays_o, self.all_rays_d, self.near, self.far], 1
            ).float()

            if len(self.all_rgbs) > 0:
                self.all_rgbs = torch.cat(self.all_rgbs, 0)

    def __len__(self):
        if self.split == "train":
            return len(self.all_rays)
        elif self.split == "val" or self.split == "test":
            return len(self.split_ids)

    def __getitem__(self, idx):
        if self.split == "train":
            sample = {
                "rays": self.all_rays[idx],
            }
            if self.get_rgb:
                sample["rgbs"] = self.all_rgbs[idx]

        # split of val/test
        elif self.split == "val":
            image_id = self.split_ids[idx]

            if self.get_rgb:
                img = Image.open(os.path.join(self.root_dir, f"{image_id:05d}.png"))
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
                img = self.transform(img)
                valid_mask = (img[-1] > 0).flatten()
                img = img.view(4, -1).permute(1, 0)
                # img = img[:,:3]*img[:,-1:]+(1-img[:,-1:])
                img = img[:, :3]

            camera_position = np.array(
                self.meta["camera"]["positions"][image_id : image_id + 1]
            )
            camera_quaternion = np.array(
                self.meta["camera"]["quaternions"][image_id : image_id + 1]
            )

            pose = np.array(
                from_position_and_quaternion(camera_position, camera_quaternion, False)
            )[0, :3, :4]
            # pose = np.array(from_position_and_quaternion(camera_position[idx:1+idx,:], camera_quaternion[idx:1+idx,:], False))[0,:3,:4]
            c2w = torch.FloatTensor(pose)[:3, :4]
            rays_o, rays_d = get_rays(self.directions, c2w)
            rays_o, rays_d = scale_rays(
                rays_o, rays_d, self.scene_boundaries, self.img_wh
            )

            self.near, self.far = calculate_near_and_far(rays_o, rays_d)
            rays = torch.cat([rays_o, rays_d, self.near, self.far], 1)  # (H*W, 8)

            sample = {"rays": rays.float(), "c2w": c2w}
            if self.get_rgb:
                sample["rgbs"] = img
                sample["valid_mask"] = valid_mask

        elif self.split == "test":
            image_id = self.split_ids[idx]

            camera_position = np.array(
                self.meta["camera"]["positions"][image_id : image_id + 1]
            )
            camera_quaternion = np.array(
                self.meta["camera"]["quaternions"][image_id : image_id + 1]
            )

            pose = np.array(
                from_position_and_quaternion(camera_position, camera_quaternion, False)
            )[0, :3, :4]
            # pose = np.array(from_position_and_quaternion(camera_position[idx:1+idx,:], camera_quaternion[idx:1+idx,:], False))[0,:3,:4]
            c2w = torch.FloatTensor(pose)[:3, :4]
            rays_o, rays_d = get_rays(self.directions, c2w)
            rays_o, rays_d = scale_rays(
                rays_o, rays_d, self.scene_boundaries, self.img_wh
            )

            self.near, self.far = calculate_near_and_far(rays_o, rays_d)
            rays = torch.cat([rays_o, rays_d, self.near, self.far], 1)  # (H*W, 8)

            sample = {"rays": rays.float(), "c2w": c2w}

        return sample
