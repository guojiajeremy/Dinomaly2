import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder
import numpy as np
import torch.multiprocessing
import json
import tifffile as tiff
import cv2
from torchvision.transforms import functional as F
from pathlib import Path
from natsort import natsorted

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    scale = (isize / size) * (isize / size)
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop((isize, isize), scale=(scale, scale), ratio=(0.95, 1.05)),
        RandomRotate90(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train, std=std_train)])
    return data_transforms


class RandomRotate90(object):
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return F.rotate(img, angle)


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        # Skip files and hidden folders such as .DS_Store or .ipynb_checkpoints.
        defect_types = [d for d in os.listdir(self.img_path)
                        if not d.startswith('.') and os.path.isdir(os.path.join(self.img_path, d))]

        for defect_type in defect_types:
            if defect_type == 'good' or defect_type == 'ok':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png") + \
                           glob.glob(os.path.join(self.gt_path, defect_type) + "/*.bmp")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = gt.convert('L')
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class MVTec3DDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, paired_transform=None, cat_3D=True, shot=None, seed=42):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')

        self.transform = transform
        self.gt_transform = gt_transform
        self.paired_transform = paired_transform
        self.phase = phase
        self.shot = shot
        self.seed = seed

        # load dataset
        self.img_paths, self.xyz_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

        # Apply few-shot sampling for training phase
        if self.phase == 'train' and self.shot is not None:
            self._apply_few_shot_sampling()

        # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0
        self.cat_3D = cat_3D

    def _apply_few_shot_sampling(self):
        """Randomly sample 'shot' number of samples from training set"""
        np.random.seed(self.seed)

        total_samples = len(self.img_paths)

        if self.shot >= total_samples:
            print(f"Warning: shot ({self.shot}) >= total samples ({total_samples}), using all samples")
            return

        # Randomly select indices
        selected_indices = np.random.choice(total_samples, size=self.shot, replace=False)
        selected_indices = np.sort(selected_indices)  # Sort to maintain some order

        # Sample the data
        self.img_paths = self.img_paths[selected_indices]
        self.xyz_paths = self.xyz_paths[selected_indices]
        self.gt_paths = self.gt_paths[selected_indices]
        self.labels = self.labels[selected_indices]
        self.types = self.types[selected_indices]

        # print(f"Few-shot sampling: selected {self.shot} samples from training set (seed={self.seed})")
        # print(f"Label distribution: good={np.sum(self.labels == 0)}, anomaly={np.sum(self.labels == 1)}")

    def load_dataset(self):
        img_tot_paths = []
        xyz_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = [d for d in os.listdir(self.img_path)
                        if not d.startswith('.') and os.path.isdir(os.path.join(self.img_path, d))]

        for defect_type in defect_types:
            if defect_type == 'good' or defect_type == 'ok':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                img_tot_paths.extend(img_paths)
                xyz_paths = [path.replace('rgb', 'z').replace('png', 'tiff') for path in img_paths]
                xyz_tot_paths.extend(xyz_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                xyz_paths = [path.replace('rgb', 'z').replace('png', 'tiff') for path in img_paths]
                xyz_tot_paths.extend(xyz_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(xyz_tot_paths), np.array(gt_tot_paths), \
            np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, xyz_path, gt, label, img_type = self.img_paths[idx], self.xyz_paths[idx], self.gt_paths[idx], \
            self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        depth = tiff.imread(xyz_path)

        depth_mask = depth > 0.
        min_value = depth[depth_mask].min()
        max_value = depth.max()
        depth = (depth - min_value) / (max_value - min_value)
        depth = 1 - depth * 0.8
        depth[depth_mask == 0] = 0

        # depth = (np.repeat(np.expand_dims(depth, axis=2), 3, axis=2) * 255.0).astype(np.uint8)
        depth = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_HOT)  # convert to obvious heatmap
        depth = depth[:, :, ::-1]

        img = np.array(img)
        img[depth_mask == 0] = 0
        img = Image.fromarray(img.astype(np.uint8))
        depth = Image.fromarray(depth.astype(np.uint8))

        if self.paired_transform is not None:
            img, depth = self.paired_transform(img, depth)
        else:
            img = self.transform(img)
            depth = self.transform(depth)

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        if self.cat_3D:
            img = torch.cat([img, depth], dim=0)

        return img, gt, label, img_path

    def cvt2heatmap(self, gray):
        heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
        return heatmap


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase, five_view_train=False):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase
        self.five_view_train = five_view_train

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels = [], [], []

        if phase == 'train':
            if self.five_view_train:
                data_set = class_json[phase]
                current_id = '0000'
                for i, sample in enumerate(data_set):
                    object_id = sample['image_path'].split('/')[:-1]
                    object_id = ''.join(object_id)
                    if object_id != current_id:
                        self.img_paths.append([])
                        self.labels.append([])
                        self.gt_paths.append([])
                        current_id = object_id
                    self.img_paths[-1].append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
                    label = False
                    self.labels[-1].append(label)
            else:
                data_set = class_json[phase]
                for sample in data_set:
                    self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
                    label = sample['anomaly_class'] != 'OK'
                    if label:
                        self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
                    else:
                        self.gt_paths.append(None)
                    self.labels.append(label)
        elif phase == 'test':
            data_set = class_json[phase]
            current_id = '0000'
            for i, sample in enumerate(data_set):
                object_id = sample['image_path'].split('/')[:-1]
                object_id = ''.join(object_id)
                if object_id != current_id:
                    self.img_paths.append([])
                    self.labels.append([])
                    self.gt_paths.append([])
                    current_id = object_id

                self.img_paths[-1].append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
                label = sample['anomaly_class'] != 'OK'
                if label:
                    self.gt_paths[-1].append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
                else:
                    self.gt_paths[-1].append(None)
                self.labels[-1].append(label)
        else:
            raise 'phase must be train or test'

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.phase == 'train':
            if self.five_view_train:
                img_paths = self.img_paths[idx]
                img_5view, gt_5view, label_5view = [], [], []
                for img_path in img_paths:
                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                    img_5view.append(img)

                img_5view = torch.stack(img_5view, dim=0)
                label = torch.ones(5, dtype=torch.uint8)
                return img_5view, label

            else:
                img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)

                return img, label
        else:
            img_paths, gts, labels = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
            img_5view, gt_5view, label_5view = [], [], []
            for img_path, gt, label in zip(img_paths, gts, labels):
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                if label == 0:
                    gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
                else:
                    gt = Image.open(gt)
                    gt = self.gt_transform(gt)
                img_5view.append(img)
                gt_5view.append(gt)
                label_5view.append(label)

            img_5view = torch.stack(img_5view, dim=0)
            gt_5view = torch.stack(gt_5view, dim=0)
            label_5view = torch.tensor(label_5view)
            return img_5view, gt_5view, label_5view, list(img_paths)


class RealIADDatasetv2(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase, five_view_train=False,
                 split='realiad_jsons', version='realiad_1024'):
        self.img_path = os.path.join(root, version, category)
        if os.path.isdir(os.path.join(self.img_path, category)):
            self.img_path = os.path.join(self.img_path, category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase
        self.five_view_train = five_view_train
        self.version = version

        json_path = os.path.join(root, split, category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels = [], [], []

        if phase == 'train':
            if self.five_view_train:
                data_set = class_json[phase]
                # 使用字典收集同一物体的所有view
                object_dict = {}
                for i, sample in enumerate(data_set):
                    object_id = sample['image_path'].split('/')[:-1]
                    object_id = ''.join(object_id)

                    if object_id not in object_dict:
                        object_dict[object_id] = {
                            'img_paths': [],
                            'labels': [],
                            'gt_paths': []
                        }

                    object_dict[object_id]['img_paths'].append(
                        os.path.join(self.img_path, sample['image_path'])
                    )
                    label = False
                    object_dict[object_id]['labels'].append(label)

                # 将字典转换为列表
                for object_id in sorted(object_dict.keys()):  # sorted保证顺序一致
                    self.img_paths.append(object_dict[object_id]['img_paths'])
                    self.labels.append(object_dict[object_id]['labels'])
                    self.gt_paths.append(object_dict[object_id]['gt_paths'])
            else:
                data_set = class_json[phase]
                for sample in data_set:
                    self.img_paths.append(os.path.join(self.img_path, sample['image_path']))
                    label = sample['anomaly_class'] != 'OK'
                    if label:
                        self.gt_paths.append(os.path.join(self.img_path, sample['mask_path']))
                    else:
                        self.gt_paths.append(None)
                    self.labels.append(label)
        elif phase == 'test':
            data_set = class_json[phase]
            # 使用字典收集同一物体的所有view
            object_dict = {}
            for i, sample in enumerate(data_set):
                object_id = sample['image_path'].split('/')[:-1]
                object_id = ''.join(object_id)

                if object_id not in object_dict:
                    object_dict[object_id] = {
                        'img_paths': [],
                        'labels': [],
                        'gt_paths': []
                    }

                object_dict[object_id]['img_paths'].append(
                    os.path.join(self.img_path, sample['image_path'])
                )

                label = sample['anomaly_class'] != 'OK'
                if sample['mask_path'] is None:
                    label = False

                if label:
                    object_dict[object_id]['gt_paths'].append(
                        os.path.join(self.img_path, sample['mask_path'])
                    )
                else:
                    object_dict[object_id]['gt_paths'].append(None)

                object_dict[object_id]['labels'].append(label)

            # 将字典转换为列表
            for object_id in sorted(object_dict.keys()):  # sorted保证顺序一致
                self.img_paths.append(object_dict[object_id]['img_paths'])
                self.labels.append(object_dict[object_id]['labels'])
                self.gt_paths.append(object_dict[object_id]['gt_paths'])
        else:
            raise ValueError('phase must be train or test')

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.phase == 'train':
            if self.five_view_train:
                img_paths = self.img_paths[idx]
                img_5view, gt_5view, label_5view = [], [], []
                for img_path in img_paths:
                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                    img_5view.append(img)

                img_5view = torch.stack(img_5view, dim=0)
                label = torch.ones(5, dtype=torch.uint8)
                return img_5view, label

            else:
                img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)

                return img, label
        else:
            img_paths, gts, labels = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
            img_5view, gt_5view, label_5view = [], [], []
            for img_path, gt, label in zip(img_paths, gts, labels):
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                if label == 0:
                    gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
                else:
                    gt = Image.open(gt)
                    gt = self.gt_transform(gt)
                img_5view.append(img)
                gt_5view.append(gt)
                label_5view.append(label)

            img_5view = torch.stack(img_5view, dim=0)
            gt_5view = torch.stack(gt_5view, dim=0)
            label_5view = torch.tensor(label_5view)
            return img_5view, gt_5view, label_5view, list(img_paths)

class DroneAnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase):
        """
        Simplified Drone Anomaly Dataset for individual images

        Args:
            root: Path to specific category (e.g., './Drone-Anomaly/Vehicle Roundabout')
            transform: Image transforms
            phase: 'train' or 'test'
        """
        self.root = Path(root)
        self.transform = transform
        self.phase = phase

        # Load dataset
        self.img_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []

        # Find all sequence directories in the category root
        sequence_dirs = [d for d in self.root.iterdir() if d.is_dir() and d.name.startswith('sequence')]

        for seq_dir in sequence_dirs:
            phase_path = seq_dir / self.phase
            if not phase_path.exists():
                continue

            if self.phase == 'train':
                # Training: all images are normal (label = 0)
                video_dirs = [d for d in phase_path.iterdir() if d.is_dir()]
                for video_dir in video_dirs:
                    img_paths = glob.glob(str(video_dir / "*.jpg"))
                    img_tot_paths.extend(img_paths)
                    tot_labels.extend([0] * len(img_paths))  # All training images are normal

            else:  # test phase
                # Testing: load images and their corresponding labels from .npy files
                video_dirs = [d for d in phase_path.iterdir() if d.is_dir()]
                for video_dir in video_dirs:
                    # Get image paths
                    img_paths = natsorted(glob.glob(str(video_dir / "*.jpg")))

                    # Load corresponding labels from .npy file
                    label_file = phase_path / f"{video_dir.name}.npy"
                    if label_file.exists():
                        try:
                            labels = np.load(label_file)
                            # Make sure we have labels for all images
                            if len(labels) == len(img_paths):
                                img_tot_paths.extend(img_paths)
                                tot_labels.extend(labels.tolist())
                            else:
                                print(
                                    f"Warning: Label count mismatch for {video_dir}: {len(labels)} labels vs {len(img_paths)} images")
                        except Exception as e:
                            print(f"Error loading labels for {video_dir}: {e}")
                            # If no labels available, assume all are normal
                            img_tot_paths.extend(img_paths)
                            tot_labels.extend([0] * len(img_paths))
                    else:
                        # If no label file, assume all are normal
                        img_tot_paths.extend(img_paths)
                        tot_labels.extend([0] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label, img_path


class MANTATinyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0
        self.phase = phase

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = [d for d in os.listdir(self.img_path)
                        if not d.startswith('.') and os.path.isdir(os.path.join(self.img_path, d))]

        for defect_type in defect_types:
            if defect_type == 'good' or defect_type == 'ok':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img_5view = [img[:, :256], img[:, 1 * 256:2 * 256], img[:, 2 * 256:3 * 256], img[:, 3 * 256:4 * 256],
                     img[:, 4 * 256:]]

        if self.phase == 'train':
            random_view_idx = random.randint(0, 4)
            img = img_5view[random_view_idx]
            img = self.transform(Image.fromarray(img))
            gt = torch.zeros([5, 1, img.shape[1], img.shape[2]])
            label = 0
            return img, label

        else:
            img_5view = [self.transform(Image.fromarray(img)) for img in img_5view]

            if label == 0:
                gt = torch.zeros([5, 1, img_5view[0].shape[1], img_5view[0].shape[2]])
            else:
                gt = Image.open(gt)
                gt = np.array(gt)
                gt_5view = [gt[:, :256], gt[:, 1 * 256:2 * 256], gt[:, 2 * 256:3 * 256], gt[:, 3 * 256:4 * 256],
                            gt[:, 4 * 256:]]
                gt_5view = [self.gt_transform(Image.fromarray(gt)) for gt in gt_5view]
                gt = torch.stack(gt_5view, dim=0)

            img = torch.stack(img_5view, dim=0)
            label = gt.flatten(1).max(dim=1)[0].int()
            return img, gt, label, img_path



class MiniDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):

        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = [d for d in os.listdir(self.img_path)
                        if not d.startswith('.') and os.path.isdir(os.path.join(self.img_path, d))]

        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
            img_tot_paths.extend(img_paths)
            tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path, label = self.img_paths[idx], self.labels[idx]
            img = Image.open(img_path).convert('RGB')
        except:
            img_path, label = self.img_paths[idx - 1], self.labels[idx - 1]
            img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label

