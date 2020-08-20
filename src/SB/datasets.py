import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src import paths


class SB_Detection(Dataset):
    TRAIN = 0
    TEST = 1

    TENSOR_LIB_NP = 0
    TENSOR_LIB_TORCH = 1

    OBJECT_COLORS = {
        'arena_center': (1, 0.8, 0.73),
        'robot': (1, 1, 0),
        'red_coin': (1, 0, 0),
        'green_coin': (0, 1, 0)
    }

    OBJECT_ID_TO_NAME = {
        0: 'arena_center',
        1: 'robot',
        2: 'red_coin',
        3: 'green_coin'
    }

    OBJECT_NAME_TO_ID = {
        'arena_center': 0,
        'robot': 1,
        'red_coin': 2,
        'green_coin': 3
    }

    def __init__(self, dataset_type=TRAIN, tensor_library=TENSOR_LIB_NP):
        if dataset_type == SB_Detection.TRAIN:
            self.dataset_folder_path = os.path.join(paths.sb_data_folder_path, 'train')
        elif dataset_type == SB_Detection.TEST:
            self.dataset_folder_path = os.path.join(paths.sb_data_folder_path, 'test')
        else:
            print('Quitting: Invalid Dataset Type Options')
            sys.exit(0)

        self.tensor_library = tensor_library

        self.frames_folder_path = os.path.join(self.dataset_folder_path, 'frames')
        self.annotations_folder_path = os.path.join(self.dataset_folder_path, 'frames annotations')
        self.annotations_file_path = os.path.join(self.annotations_folder_path, 'annotations.json')

        # Checking if frames folder exists
        if not os.path.isdir(self.frames_folder_path):
            print('Quitting: Source folder "frames" does not exist')
            sys.exit(0)

        # Checking if frames annotations folder exists
        if not os.path.isdir(self.annotations_folder_path):
            print('Quitting: Frames folder "frames" does not exist')
            sys.exit(0)

        with open(self.annotations_file_path) as annotations_file:
            self.annotations = json.load(annotations_file)

        self.num_instances = len(self.annotations)

    def __len__(self):
        return self.num_instances

    def __getitem__(self, idx):
        instance = self.annotations[idx]

        # Reading image, swapping channels from BGR to RGB and normalizing it tp [0, 1]
        image = cv2.cvtColor(
            cv2.imread(
                os.path.join(self.frames_folder_path, instance['frame_file_name']), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        ) / 255
        labels = []
        bounding_boxes = []
        for object in instance['objects'].keys():
            for bounding_box in instance['objects'][object]['bounding_boxes']:
                labels.append(instance['objects'][object]['id'])
                bounding_boxes.append(bounding_box['tl'] + bounding_box['br'])

        if self.tensor_library == SB_Detection.TENSOR_LIB_NP:
            input = np.asarray(image, dtype=np.float32)

            output = {}
            output['boxes'] = np.asarray(bounding_boxes, dtype=np.float32)
            output['labels'] = np.asarray(labels, dtype=np.int64)
        else:
            input = torch.as_tensor(SB_Detection.bring_image_channels_first(image), dtype=torch.float32)

            output = {}
            output['boxes'] = torch.as_tensor(bounding_boxes, dtype=torch.float32)
            output['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        return input, output

    @staticmethod
    def bring_image_channels_first(image):
        assert type(image) == np.ndarray
        return np.transpose(image, axes=(2, 0, 1))

    @staticmethod
    def send_image_channels_back(image):
        assert type(image) == np.ndarray
        return np.transpose(image, axes=(1, 2, 0))

    @staticmethod
    def collate_dataloader_batch(batch):
        return tuple(zip(*batch))

    def shuffle(self):
        np.random.shuffle(self.annotations)
