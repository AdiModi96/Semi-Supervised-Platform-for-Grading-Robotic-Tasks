import json
import os

import cv2
from bs4 import BeautifulSoup as BS4
from tqdm import tqdm

from src import paths
from src.SB.utils import VideoSurfer as VS

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


def extract_frames_and_get_annotations(video_file_path, video_annotations_file_path):
    file_prefix = video_file_path.split(os.sep)[-1].replace('.mp4', '')
    frames_annotations = []

    video_surfer = VS(video_file_path)
    progress_bar = tqdm(total=len(video_surfer))
    with open(video_annotations_file_path) as video_annotations_file:

        video_annotations_xml = BS4(video_annotations_file.read(), 'lxml')
        for frame_idx in range(len(video_surfer)):
            frame = video_surfer[frame_idx]
            frame_file_name = file_prefix + '_frame_' + str(frame_idx).zfill(5) + '.jpg'
            cv2.imwrite(os.path.join(paths.sb_data_folder_path, TYPE, 'frames', frame_file_name), frame)

            frame_annotations = {
                'frame_file_name': frame_file_name,
                'objects': {
                    'arena_center': {
                        'id': OBJECT_NAME_TO_ID['arena_center'],
                        'bounding_boxes': []
                    },
                    'robot': {
                        'id': OBJECT_NAME_TO_ID['robot'],
                        'bounding_boxes': []
                    },
                    'red_coin': {
                        'id': OBJECT_NAME_TO_ID['red_coin'],
                        'bounding_boxes': []
                    },
                    'green_coin': {
                        'id': OBJECT_NAME_TO_ID['green_coin'],
                        'bounding_boxes': []
                    }
                }
            }

            for box in video_annotations_xml.find_all('box', {'frame': frame_idx}):
                frame_annotations['objects'][box.parent['label']]['bounding_boxes'].append(
                    {'tl': [float(box['xtl']), float(box['ytl'])], 'br': [float(box['xbr']), float(box['ybr'])]}
                )

            frames_annotations.append(frame_annotations)
            progress_bar.update(1)

        progress_bar.close()

        return frames_annotations


TYPE = 'train'
# TYPE = 'test'

# Checking if videos folder exists
videos_folder_path = os.path.join(paths.sb_data_folder_path, TYPE, 'videos')
if not os.path.exists(videos_folder_path):
    print('Quitting: Source "videos" folder does not exist')

# Checking if video annotations folder exists
videos_annotations_folder_path = os.path.join(paths.sb_data_folder_path, TYPE, 'videos annotations')
if not os.path.isdir(videos_annotations_folder_path):
    print('Quitting: Source "videos annotations" folder does not exist')

# Checking and creating frames folder
frames_folder_path = os.path.join(paths.sb_data_folder_path, TYPE, 'frames')
if not os.path.isdir(frames_folder_path):
    os.makedirs(frames_folder_path)

# Master list to have annotations from all frames from all video files
frames_annotations = []

# Iterating through video files
for file_name in os.listdir(videos_folder_path):

    # Finding video file path and the corresponding annotations file path 
    video_file_path = os.path.join(videos_folder_path, file_name)
    if not os.path.isfile(video_file_path):
        continue

    video_annotations_file_path = os.path.join(videos_annotations_folder_path, file_name.replace('.mp4', '.xml'))
    if os.path.isfile(video_annotations_file_path):
        print('Processing video: {}...'.format(file_name))
    else:
        print('Skipping: annotations does not exist for video: {}'.format(file_name))

    frames_annotations += extract_frames_and_get_annotations(video_file_path, video_annotations_file_path)
    break

frames_annotations_folder_path = os.path.join(paths.sb_data_folder_path, TYPE, 'frames annotations')
if not os.path.isdir(frames_annotations_folder_path):
    os.makedirs(frames_annotations_folder_path)

frames_annotations_file_path = os.path.join(frames_annotations_folder_path, 'annotations.json')
with open(frames_annotations_file_path, 'w') as frames_annotations_file:
    json.dump(frames_annotations, frames_annotations_file, indent=4)
