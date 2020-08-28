import math
import os
import re
import time

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import paths
from src.SB.datasets import SB_Detection
from src.SB.models import FasterRCNN


def train_FasterRCNN(hyperparameters):
    if hyperparameters['device'] == 'cuda':
        torch.cuda.init()

    # Initializing network
    network = FasterRCNN(pretrained=hyperparameters['network']['pretrained'])

    if hyperparameters['network']['weights_file_path'] and os.path.isfile(hyperparameters['network']['weights_file_path']):
        network.set_state_dict(torch.load(hyperparameters['network']['weights_file_path']))

    network.train()
    network.to(hyperparameters['device'])

    # Initializing dataset
    dataset = SB_Detection(
        dataset_type=SB_Detection.TRAIN,
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

    # Defining optimizer
    optimizer = hyperparameters['optimizer']['type'](
        network.parameters(),
        lr=hyperparameters['optimizer']['learning_rate'],
        betas=hyperparameters['optimizer']['betas'],
        weight_decay=hyperparameters['optimizer']['weight_decay']
    )

    # Finding training instance folder path
    instance = 0
    trained_model_weights_folder_path = os.path.join(paths.trained_models_weights_folder_path, network.__class__.__name__)
    while os.path.isdir(os.path.join(trained_model_weights_folder_path, 'Instance_' + str(instance).zfill(3))):
        instance += 1
    training_instance_folder_path = os.path.join(trained_model_weights_folder_path, 'Instance_' + str(instance).zfill(3))

    if not os.path.isdir(training_instance_folder_path):
        os.makedirs(training_instance_folder_path)

    if hyperparameters['network']['weights_file_path']:
        file_name = hyperparameters['network']['weights_file_path'].split(os.sep)[-1]
        loss_threshold = float(re.findall('[0-9].[0-9]+', file_name)[0])
    else:
        loss_threshold = float('inf')

    for epoch_idx in range(hyperparameters['num_epochs']):

        print('-' * 80)
        print('Epoch: {} of {}...'.format(epoch_idx + 1, hyperparameters['num_epochs']))
        print('-' * 80)

        batch_idx = 0

        total_batches = math.ceil(len(dataset) / hyperparameters['batch_size'])
        progress_bar = tqdm(desc='Losses: ', total=total_batches)
        sum_batch_losses = float('inf')
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
            inputs = [image.to(hyperparameters['device']) for image in inputs]
            outputs = [{key: output[key].to(hyperparameters['device']) for key in output.keys()} for output in outputs]

            # Getting batch loss
            batch_loss_dictionary = network.fit_batch(inputs, outputs)
            sum_batch_losses = sum(loss for loss in batch_loss_dictionary.values())

            # Back-propagation
            optimizer.zero_grad()
            sum_batch_losses.backward()
            optimizer.step()

            # Calculating epoch Loss
            for loss_type in batch_loss_dictionary.keys():
                epoch_loss_dictionary[loss_type] += len(batch) * batch_loss_dictionary[loss_type].item()

            if sum_batch_losses <= loss_threshold:
                loss_threshold = sum_batch_losses / 1.1
                torch.save(
                    network.get_state_dict(),
                    os.path.join(
                        training_instance_folder_path, 'Epoch-{} -- Batch-{} -- Batch Loss-{}.pt'.format(
                            str(epoch_idx + 1).zfill(3), str(batch_idx + 1).zfill(5), round(sum_batch_losses.item(), 7)
                        )
                    )
                )

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
            break

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

        torch.save(
            network.get_state_dict(),
            os.path.join(
                training_instance_folder_path, 'Epoch-{} -- Epoch Loss-{}.pt'.format(
                    str(epoch_idx + 1).zfill(3), round(sum_epoch_losses, 7))
            )
        )

        print('-' * 80)
        print()


if __name__ == '__main__':
    # Defining hyperparameters for FasterRCNN
    hyperparameters = {
        'batch_size': 5,
        'num_workers': 5,
        'num_epochs': 30,
        'device': 'cpu',
        'optimizer': {
            'type': optim.Adam,
            'learning_rate': 1e-4,
            'betas': (0.9, 0.99),
            'weight_decay': 0.0005
        },
        'network': {
            'pretrained': True,
            'weights_file_path': None
        }
    }
    if torch.cuda.is_available():
        hyperparameters['device'] = 'cuda'

    print('Commencing Training')
    train_FasterRCNN(hyperparameters)
    print('Training Completed')
