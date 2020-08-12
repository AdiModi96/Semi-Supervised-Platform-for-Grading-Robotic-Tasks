import random

from matplotlib import pyplot as plt

from src.SB.datasets import SB_Detection
from src.SB.utils import ImageUtils as IU


def test_SB_dataset(dataset_type=SB_Detection.TRAIN):
    dataset = SB_Detection(dataset_type=dataset_type, tensor_library=SB_Detection.TENSOR_LIB_NP)

    input, output = dataset[random.randint(0, len(dataset))]
    image = IU.draw_annotation(image=input, annotations=output)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.show()


if __name__ == '__main__':
    test_SB_dataset(dataset_type=SB_Detection.TRAIN)
