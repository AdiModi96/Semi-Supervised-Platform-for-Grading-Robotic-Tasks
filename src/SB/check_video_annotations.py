import os
import sys

import cv2
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.SB.utils import ImageUtils as IU

OBJECT_ID_TO_NAME = {
    1: 'arena_center',
    2: 'robot',
    3: 'red_coin',
    4: 'green_coin'
}

OBJECT_NAME_TO_ID = {
    'arena_center': 1,
    'robot': 2,
    'red_coin': 3,
    'green_coin': 4
}


def play_annotation(video_file_paths, annotation_file_paths, playback_speed=1.0):
    for video_file_path, annotation_file_path in zip(video_file_paths, annotation_file_paths):

        if not os.path.exists(video_file_path):
            print('Video file does not exist, check file path')
            sys.exit(0)

        if not os.path.exists(annotation_file_path):
            print('Annotation file does not exist, check file path')
            sys.exit(0)

        video_file_name = video_file_path.split(os.sep)[-1]
        print('Checking annotations for file: {}'.format(video_file_name))

        video = cv2.VideoCapture(video_file_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        with open(annotation_file_path) as file:
            video_annotations_xml = BeautifulSoup(file.read(), 'lxml')

        frame_annotations = []
        parsing_progress_bar = tqdm(total=num_frames, unit='frames')
        parsing_progress_bar.set_description('Parsing annotations')
        for frame_idx in range(num_frames):

            annotations = {
                'boxes': [],
                'labels': []
            }

            for box in video_annotations_xml.find_all('box', {'frame': frame_idx}):
                annotations['boxes'].append(
                    (float(box['xtl']), float(box['ytl']), float(box['xbr']), float(box['ybr']))
                )
                annotations['labels'].append(OBJECT_NAME_TO_ID[box.parent['label']])

            frame_annotations.append(annotations)

            parsing_progress_bar.update(1)

        parsing_progress_bar.close()

        input('Annotations parsing complete, press enter to continue ...')

        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_idx in range(num_frames):
            frame = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2RGB) / 255
            annotated_frame = cv2.cvtColor(
                np.asarray(IU.draw_annotation(frame, frame_annotations[frame_idx]) * 255, dtype=np.uint8),
                cv2.COLOR_RGB2BGR
            )
            cv2.imshow(video_file_name, annotated_frame)
            cv2.waitKey(int(1000 / (playback_speed * fps)))

        cv2.destroyAllWindows()


video_file_paths = [os.path.abspath(r"")]
annotation_file_paths = [os.path.abspath(r"")]
playback_speed = 2.0
play_annotation(video_file_paths, annotation_file_paths, playback_speed)
