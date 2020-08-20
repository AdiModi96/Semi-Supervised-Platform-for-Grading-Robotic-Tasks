import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import paths
from src.SB.datasets import SB_Detection
from src.SB.models import FasterRCNN


def get_losses(hyperparameters):
    if hyperparameters['device'] == 'cuda':
        torch.cuda.init()

    # Initializing network
    network = FasterRCNN()
    if hyperparameters['network']['weights_file_path']:
        network.set_state_dict(torch.load(hyperparameters['network']['weights_file_path']))
    network.train()
    network.to(hyperparameters['device'])

    # Initializing dataset
    dataset = SB_Detection(
        dataset_type=hyperparameters['dataset_type'],
        tensor_library=SB_Detection.TENSOR_LIB_TORCH
    )

    dataset.shuffle()
    dataloader = DataLoader(
        dataset,
        batch_size=hyperparameters['batch_size'],
        num_workers=hyperparameters['num_workers'],
        collate_fn=SB_Detection.collate_dataloader_batch,
        shuffle=True
    )

    print('-' * 80)

    batch_idx = 0

    total_batches = math.ceil(len(dataset) / hyperparameters['batch_size'])
    progress_bar = tqdm(desc='Batches Completed', total=total_batches)
    epoch_loss_dictionary = {
        'loss_rpn_box_reg': 0,
        'loss_objectness': 0,
        'loss_classifier': 0,
        'loss_box_reg': 0,
    }
    for batch in dataloader:

        # Getting network inputs and outputs for the batch
        inputs, outputs = batch
        inputs = [image.to(hyperparameters['device']) for image in inputs]
        outputs = [{key: output[key].to(hyperparameters['device']) for key in output.keys()} for output in outputs]

        # Getting batch loss
        batch_loss_dictionary = network.fit_batch(inputs, outputs)

        # Calculating epoch Loss
        for loss_type in batch_loss_dictionary.keys():
            epoch_loss_dictionary[loss_type] += len(batch) * batch_loss_dictionary[loss_type].item()

        progress_bar.set_description(
            'Losses: RPN Regression: {} | RPN Objecteness: {} | Classification: {} | Regression: {}'.format(
                round(batch_loss_dictionary['loss_rpn_box_reg'].item(), 7),
                round(batch_loss_dictionary['loss_objectness'].item(), 7),
                round(batch_loss_dictionary['loss_classifier'].item(), 7),
                round(batch_loss_dictionary['loss_box_reg'].item(), 7)
            )
        )
        batch_idx += 1
        progress_bar.update(1)

    print('-' * 80)

    print('Epoch Losses: ')
    print('•RPN Regression:\n\t•Cumulative: {}\n\t•Mean: {}'.format(round(epoch_loss_dictionary['loss_rpn_box_reg'], 7), round(epoch_loss_dictionary['loss_rpn_box_reg'], 7) / len(dataset)))
    print('•RPN Objectness:\n\t•Cumulative: {}\n\t•Mean: {}'.format(round(epoch_loss_dictionary['loss_objectness'], 7), round(epoch_loss_dictionary['loss_objectness'], 7) / len(dataset)))
    print('•Classification:\n\t•Cumulative: {}\n\t•Mean: {}'.format(round(epoch_loss_dictionary['loss_classifier'], 7), round(epoch_loss_dictionary['loss_classifier'], 7) / len(dataset)))
    print('•Regression:\n\t•Cumulative: {}\n\t•Mean: {}'.format(round(epoch_loss_dictionary['loss_box_reg'], 7), round(epoch_loss_dictionary['loss_box_reg'], 7) / len(dataset)))

    print('-' * 80)
    print()


if __name__ == '__main__':
    # Defining hyperparameters for FasterRCNN
    hyperparameters = {
        'batch_size': 5,
        'num_workers': 5,
        'device': 'cpu',
        'dataset_type': SB_Detection.TEST,
        'network': {
            'weights_file_path': os.path.join(paths.trained_models_weights_folder_path, 'FasterRCNN', 'Instance_000', 'Epoch-1 -- Batch-1 -- Batch Loss-2.4029379.pt')
        }
    }
    if torch.cuda.is_available():
        hyperparameters['device'] = 'cuda'

    print('Commencing Evaluation')
    get_losses(hyperparameters)
    print('Evaluation Completed')
