import math
import os
import re
import time

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import paths
from src.SB.datasets import SB_Detection
from src.SB.models import FasterRCNN


def train_FasterRCNN(hyperparameters):
    if hyperparameters['device'] == 'cuda':
        torch.cuda.init()

    # Initializing network
    network = FasterRCNN(pretrained=hyperparameters['network']['pretrained'])
    if hyperparameters['network']['weights_file_path']:
        network.set_state_dict(torch.load(hyperparameters['network']['weights_file_path']))
    network.train()
    network.to(hyperparameters['device'])

    # Initializing dataset
    dataset = SB_Detection(dataset_type=SB_Detection.TRAIN, tensor_library=SB_Detection.TENSOR_LIB_TORCH)
    dataset.shuffle()
    dataloader = DataLoader(dataset,
                            batch_size=hyperparameters['batch_size'],
                            num_workers=hyperparameters['num_workers'],
                            collate_fn=SB_Detection.collate_dataloader_batch,
                            shuffle=True)

    # Defining optimizer
    optimizer = hyperparameters['optimizer']['type'](network.parameters(),
                                                     lr=hyperparameters['optimizer']['learning_rate'],
                                                     betas=hyperparameters['optimizer']['betas'],
                                                     weight_decay=hyperparameters['optimizer']['weight_decay'])

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
        epoch_start_time = time.time()

        print('-' * 80)
        print('Epoch: {} of {}...'.format(epoch_idx + 1, hyperparameters['num_epochs']))
        print('-' * 80)

        batch_idx = 0
        total_batches = math.ceil(len(dataset) / hyperparameters['batch_size'])
        progress_bar = tqdm(desc='Batches Completed', total=total_batches)
        batch_loss = float('inf')
        epoch_loss_dict = {}
        for batch in dataloader:

            inputs, outputs = batch
            inputs = [image.to(hyperparameters['device']) for image in inputs]
            outputs = [{key: output[key].to(hyperparameters['device']) for key in output.keys()} for output in outputs]

            loss_dict = network.fit_batch(inputs, outputs)
            batch_loss = sum(loss for loss in loss_dict.values())
            for loss_type in loss_dict.keys():
                if loss_type in epoch_loss_dict.keys():
                    epoch_loss_dict[loss_type] += loss_dict[loss_type].item()
                else:
                    epoch_loss_dict[loss_type] = loss_dict[loss_type].item()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if batch_loss <= loss_threshold:
                loss_threshold = 0.80 * batch_loss
                torch.save(network.get_state_dict(),
                           os.path.join(training_instance_folder_path, 'Loss-{}.pt'.format(round(batch_loss.item(), 7))))

            batch_idx += 1
            progress_bar.set_description(
                'Losses: RPN Regression: {} | RPN Objecteness: {} | Classification: {} | Regression: {}'.format(
                    round(loss_dict['loss_rpn_box_reg'].item(), 5),
                    round(loss_dict['loss_objectness'].item(), 5),
                    round(loss_dict['loss_classifier'].item(), 5),
                    round(loss_dict['loss_box_reg'].item(), 5)
                ))
            progress_bar.update(1)

        epoch_end_time = time.time()
        progress_bar.close()

        torch.save(network.get_state_dict(),
                   os.path.join(training_instance_folder_path, 'Epoch-{} Loss-{}.pt'.format(epoch_idx + 1, batch_loss)))

        print('-' * 80)

        print('Epoch Losses: ')
        print('•RPN Regression:\n\t•Cumulative: {}\n\t•Mean:{}'.format(round(epoch_loss_dict['loss_rpn_box_reg'], 10), round(epoch_loss_dict['loss_rpn_box_reg'], 10) / len(dataset)))
        print('•RPN Objectness:\n\t•Cumulative: {}\n\t•Mean:{}'.format(round(epoch_loss_dict['loss_objectness'], 10), round(epoch_loss_dict['loss_rpn_box_reg'], 10) / len(dataset)))
        print('•Classification:\n\t•Cumulative: {}\n\t•Mean:{}'.format(round(epoch_loss_dict['loss_classifier'], 10), round(epoch_loss_dict['loss_rpn_box_reg'], 10) / len(dataset)))
        print('•Regression:\n\t•Cumulative: {}\n\t•Mean:{}'.format(round(epoch_loss_dict['loss_box_reg'], 10), round(epoch_loss_dict['loss_rpn_box_reg'], 10) / len(dataset)))

        print('Epoch Time:', epoch_end_time - epoch_start_time)
        print('-' * 80)
        print()


if __name__ == '__main__':
    # Defining hyperparameters for FasterRCNN
    hyperparameters = {
        'batch_size': 2,
        'num_workers': 5,
        'num_epochs': 30,
        'device': 'cpu',
        'optimizer': {
            'type': optim.Adam,
            'learning_rate': 1e-6,
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
