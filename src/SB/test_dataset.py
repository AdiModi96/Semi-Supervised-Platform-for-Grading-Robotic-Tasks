import numpy as np
from matplotlib import pyplot as plt

from src.SB.datasets import SB_Detection
from src.SB.utils import ImageUtils as IU


def test_SB_Detection_dataset(dataset_type=SB_Detection.TRAIN):
    dataset = SB_Detection(dataset_type=dataset_type, tensor_library=SB_Detection.TENSOR_LIB_NP)

    dataset.shuffle()

    num_subplots = (2, 2)
    subplot_idx = 1

    plt.figure(num='Images')
    for instance_idx in np.random.randint(0, len(dataset), size=(num_subplots[0] * num_subplots[1])):
        input, output = dataset[instance_idx]

        image = IU.draw_annotation(image=input, annotations=output)

        plt.subplot(num_subplots[0], num_subplots[1], subplot_idx)
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        subplot_idx += 1
    plt.show()


if __name__ == '__main__':
    test_SB_Detection_dataset(dataset_type=SB_Detection.TEST)
