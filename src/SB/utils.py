import os

import cv2

from src.SB.datasets import SB_Detection


class VideoSurfer:

    def __init__(self, video_file_path):
        super().__init__()
        self.video_file_path = video_file_path
        if not os.path.exists(self.video_file_path):
            raise IOError('Video file does not exists!')

        self.video = cv2.VideoCapture(self.video_file_path)
        self.num_frames = int(self.video.get(propId=cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        idx = min(self.num_frames - 1, idx)
        self.video.set(propId=cv2.CAP_PROP_POS_FRAMES, value=idx)
        return self.video.read()[1]

    def __del__(self):
        self.video.release()


class ImageUtils:

    def __init__(self) -> None:
        super().__init__()

    # If weights are none, all of the images are merges with equal weights
    @staticmethod
    def draw_annotation(image, annotations):
        image = image.copy()
        boxes = annotations['boxes']
        labels = annotations['labels']
        for i in range(len(labels)):
            point_tl = (int(boxes[i][0]), int(boxes[i][1]))
            point_br = (int(boxes[i][2]), int(boxes[i][3]))

            cv2.rectangle(
                image,
                pt1=point_tl,
                pt2=point_br,
                color=SB_Detection.OBJECT_COLORS[SB_Detection.OBJECT_ID_TO_NAME[labels[i]]],
                thickness=2
            )
            cv2.putText(
                image,
                text=SB_Detection.OBJECT_ID_TO_NAME[labels[i]],
                org=(point_tl[0], point_tl[1] - 2),
                fontFace=cv2.FONT_ITALIC,
                fontScale=0.75,
                color=SB_Detection.OBJECT_COLORS[SB_Detection.OBJECT_ID_TO_NAME[labels[i]]],
                thickness=1
            )

        return image
