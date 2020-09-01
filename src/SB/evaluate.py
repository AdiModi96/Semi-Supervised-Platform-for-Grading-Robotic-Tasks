import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import paths
from src.SB.datasets import SB_Detection
from src.SB.models import FasterRCNN


def get_losses(parameters):
    if parameters['device'] == 'cuda':
        torch.cuda.init()

    # Initializing network
    network = parameters['network']['model']()
    if os.path.isfile(parameters['network']['weights_file_path']):
        network.set_state_dict(torch.load(parameters['network']['weights_file_path']))
    else:
        print('Quitting: Network weight\'s file does not exists')
        sys.exit(0)

    network.train()
    network.to(parameters['device'])

    # Initializing dataset
    dataset = SB_Detection(
        dataset_type=parameters['dataset_type'],
        tensor_library=SB_Detection.TENSOR_LIB_TORCH
    )

    dataset.shuffle()
    dataloader = DataLoader(
        dataset,
        batch_size=parameters['batch_size'],
        num_workers=parameters['num_workers'],
        collate_fn=SB_Detection.collate_dataloader_batch,
        shuffle=True
    )

    print('-' * 80)

    batch_idx = 0

    total_batches = math.ceil(len(dataset) / parameters['batch_size'])
    progress_bar = tqdm(desc='Batches Completed', total=total_batches)
    epoch_loss_dictionary = {
        'loss_rpn_box_reg': 0,
        'loss_objectness': 0,
        'loss_classifier': 0,
        'loss_box_reg': 0,
    }
    epoch_start_time = time.time()
    for batch in dataloader:

        # Getting network inputs and outputs for the batch
        inputs, outputs = batch
        inputs = [image.to(parameters['device']) for image in inputs]
        outputs = [{key: output[key].to(parameters['device']) for key in output.keys()} for output in outputs]

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

    epoch_end_time = time.time()
    progress_bar.close()

    sum_epoch_losses = sum([(epoch_loss_dictionary[key] / len(dataset)) for key in epoch_loss_dictionary.keys()])

    print('-' * 80)

    epoch_summary = '''
            Epoch Losses:
            •RPN Regression:
                •Cumulative: {}
                •Mean: {}
            •RPN Objectness:
                •Cumulative: {}
                •Mean: {}
            •Classification:
                •Cumulative: {}
                •Mean: {}
            •Regression:
                •Cumulative: {}
                •Mean: {}
            •Sum Epoch Losses:
                •Mean: {}
            Epoch Time: {}
            '''.format(
        round(epoch_loss_dictionary['loss_rpn_box_reg'], 7), round(epoch_loss_dictionary['loss_rpn_box_reg'] / len(dataset), 7),
        round(epoch_loss_dictionary['loss_objectness'], 7), round(epoch_loss_dictionary['loss_objectness'] / len(dataset), 7),
        round(epoch_loss_dictionary['loss_classifier'], 7), round(epoch_loss_dictionary['loss_classifier'] / len(dataset), 7),
        round(epoch_loss_dictionary['loss_box_reg'], 7), round(epoch_loss_dictionary['loss_box_reg'] / len(dataset), 7),
        sum_epoch_losses,
        round(epoch_end_time - epoch_start_time, 3)
    )

    print(epoch_summary)


if __name__ == '__main__':
    # Building parameters for evaluation
    parameters = {
        'batch_size': 3,
        'num_workers': 3,
        'device': 'cpu',
        'dataset_type': SB_Detection.TRAIN,
        'network': {
            'model': FasterRCNN,
            'weights_file_path': os.path.join(paths.trained_models_weights_folder_path, 'FasterRCNN', 'Instance_005', 'Epoch-2 -- Epoch Loss-0.0012627.pt')
        }
    }
    if torch.cuda.is_available():
        parameters['device'] = 'cuda'

    print('Commencing Evaluation')
    get_losses(parameters)
    print('Evaluation Completed')
