import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt

from src import paths
from src.SB.datasets import SB_Detection
from src.SB.models import FasterRCNN
from src.SB.utils import ImageUtils as IU


def test_FasterRCNN():
    # Initializing dataset
    dataset = SB_Detection(dataset_type=SB_Detection.TEST, tensor_library=SB_Detection.TENSOR_LIB_TORCH)

    dataset.shuffle()
    network = FasterRCNN()

    trained_model_file_path = os.path.join(paths.trained_models_weights_folder_path, 'FasterRCNN', 'Instance_001', 'Loss-0.0226141.pt')
    print('Testing using {}'.format(trained_model_file_path))
    network.set_state_dict(torch.load(os.path.join(trained_model_file_path)))
    network.eval()

    annotated_image, target = dataset[random.randint(0, len(dataset))]

    predicted_annotations = network.predict_batch(torch.unsqueeze(annotated_image, dim=0))[0]
    predicted_annotations['boxes'] = predicted_annotations['boxes'].to('cpu').detach().numpy()
    predicted_annotations['labels'] = predicted_annotations['labels'].to('cpu').detach().numpy()
    predicted_annotations['scores'] = predicted_annotations['scores'].to('cpu').detach().numpy()

    # Filtering
    # condition = predicted_annotations['scores'] > 0.8
    # predicted_annotations['boxes'] = predicted_annotations['boxes'][condition]
    # predicted_annotations['labels'] = predicted_annotations['labels'][condition]
    # predicted_annotations['scores'] = predicted_annotations['scores'][condition]

    annotated_image = SB_Detection.send_image_channels_back(annotated_image.to('cpu').detach().numpy())

    annotated_image = IU.draw_annotation(annotated_image, predicted_annotations)

    annotated_image = np.asarray(annotated_image * 255, dtype=np.uint8)
    plt.figure(num='Predicted Annotations')
    plt.imshow(annotated_image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    print('Commencing Testing')
    for i in range(10):
        test_FasterRCNN()
    print('Testing Completed')
