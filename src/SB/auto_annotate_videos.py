import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src import paths
from src.SB.datasets import SB_Detection
from src.SB.models import FasterRCNN
from src.SB.utils import FrameExtractor as FE, ImageUtils as IU


def auto_annotate(parameters):
    if parameters['device'] == 'cuda':
        torch.cuda.init()

    network = parameters['network']['model']()
    if parameters['network']['weights_file_path'] and os.path.isfile(parameters['network']['weights_file_path']):
        network.set_state_dict(torch.load(parameters['network']['weights_file_path']))
    else:
        print('Quitting: Network weight\'s file does not exists')
        sys.exit(0)

    network.eval()
    network.to(parameters['device'])

    for video_file_path in parameters['video_file_paths']:
        if not os.path.isfile(video_file_path):
            print('Quitting: Video file does not exists')
            sys.exit(0)

        print('-' * 80)
        print('Annotating video: {}'.format(video_file_path))
        print('-' * 80)
        frame_extractor = FE(video_file_path=video_file_path)

        annotated_video_folder_path = os.path.join(paths.auto_annotation_videos_folder_path, network.__class__.__name__)
        if not os.path.isdir(annotated_video_folder_path):
            os.makedirs(annotated_video_folder_path)

        annotated_video_file_path = os.path.join(annotated_video_folder_path, 'annotated_' + video_file_path.split(os.sep)[-1])

        annotated_video_writer = cv2.VideoWriter(
            annotated_video_file_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            frame_extractor.fps,
            frame_extractor.frame_shape
        )

        progress_bar = tqdm(total=len(frame_extractor))
        for frame_idx in range(len(frame_extractor)):
            frame = cv2.cvtColor(frame_extractor[frame_idx], cv2.COLOR_BGR2RGB) / 255

            predicted_annotations = network.predict_batch(
                torch.unsqueeze(
                    torch.as_tensor(SB_Detection.bring_image_channels_first(frame), dtype=torch.float32),
                    dim=0
                ).to(parameters['device'])
            )[0]

            predicted_annotations['boxes'] = predicted_annotations['boxes'].to('cpu').detach().numpy()
            predicted_annotations['labels'] = predicted_annotations['labels'].to('cpu').detach().numpy()
            predicted_annotations['scores'] = predicted_annotations['scores'].to('cpu').detach().numpy()

            # Filtering
            condition = predicted_annotations['scores'] > 0.95
            predicted_annotations['boxes'] = predicted_annotations['boxes'][condition]
            predicted_annotations['labels'] = predicted_annotations['labels'][condition]
            predicted_annotations['scores'] = predicted_annotations['scores'][condition]

            # Annotating that Frame
            annotated_frame = IU.draw_annotation(frame, predicted_annotations) * 255
            annotated_frame = cv2.cvtColor(np.asarray(annotated_frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            annotated_video_writer.write(annotated_frame)

            progress_bar.update(1)
        progress_bar.close()
        annotated_video_writer.release()

        print('-' * 80)
        print('Annotated video saved at: {}'.format(annotated_video_file_path))
        print('-' * 80)
        print()


if __name__ == '__main__':
    print('Commencing Annotation')
    parameters = {
        'video_file_paths': [
            # os.path.join(r'D:\Codes\Python\IIT Bombay\Semester 4\[CS 694] Seminar\data\SB\03 - Cropped\1033_original.mp4       ')
            os.path.join(r'D:\Codes\Python\IIT Bombay\Semester 4\[CS 694] Seminar\data\SB\03 - Cropped', video_file_name) for video_file_name in os.listdir(os.path.join(r'D:\Codes\Python\IIT Bombay\Semester 4\[CS 694] Seminar\data\SB\03 - Cropped'))
        ],
        'device': 'cpu',
        'network': {
            'model': FasterRCNN,
            'weights_file_path': os.path.join(paths.trained_models_weights_folder_path, 'FasterRCNN', 'Instance_003', 'Epoch-3 -- Epoch Loss-0.000976.pt')
        }
    }

    if torch.cuda.is_available():
        parameters['device'] = 'cuda'

    auto_annotate(parameters)
    print('Annotation Completed')
