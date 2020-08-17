import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.SB.datasets import SB_Detection
from src.SB.models import FasterRCNN


def get_losses(hyperparameters):
    if hyperparameters['device'] == 'cuda':
        torch.cuda.init()

    # Initializing network
    network = FasterRCNN(pretrained=hyperparameters['network']['pretrained'])
    if hyperparameters['network']['weights_file_path']:
        network.set_state_dict(torch.load(hyperparameters['network']['weights_file_path']))
    network.train()
    network.to(hyperparameters['device'])

    # Initializing dataset
    dataset = SB_Detection(dataset_type=hyperparameters['dataset_type'], tensor_library=SB_Detection.TENSOR_LIB_TORCH)
    dataset.shuffle()
    dataloader = DataLoader(dataset,
                            batch_size=hyperparameters['batch_size'],
                            num_workers=hyperparameters['num_workers'],
                            collate_fn=SB_Detection.collate_dataloader_batch,
                            shuffle=True)

    print('-' * 80)

    batch_idx = 0
    total_batches = math.ceil(len(dataset) / hyperparameters['batch_size'])
    progress_bar = tqdm(desc='Batches Completed', total=total_batches)
    epoch_loss_dict = {}
    for batch in dataloader:

        inputs, outputs = batch
        inputs = [image.to(hyperparameters['device']) for image in inputs]
        outputs = [{key: output[key].to(hyperparameters['device']) for key in output.keys()} for output in outputs]

        loss_dict = network.fit_batch(inputs, outputs)
        for loss_type in loss_dict.keys():
            if loss_type in epoch_loss_dict.keys():
                epoch_loss_dict[loss_type] += loss_dict[loss_type].item()
            else:
                epoch_loss_dict[loss_type] = loss_dict[loss_type].item()

        batch_idx += 1
        progress_bar.set_description(
            'Losses: RPN Regression: {} | RPN Objecteness: {} | Classification: {} | Regression: {}'.format(
                round(loss_dict['loss_rpn_box_reg'].item(), 5),
                round(loss_dict['loss_objectness'].item(), 5),
                round(loss_dict['loss_classifier'].item(), 5),
                round(loss_dict['loss_box_reg'].item(), 5)
            ))
        progress_bar.update(1)

    progress_bar.close()

    print('-' * 80)

    print('Epoch Losses: ')
    print('•RPN Regression:\n\t•Cumulative: {}\n\t•Mean:{}'.format(round(epoch_loss_dict['loss_rpn_box_reg'], 10), round(epoch_loss_dict['loss_rpn_box_reg'], 10) / len(dataset)))
    print('•RPN Objectness:\n\t•Cumulative: {}\n\t•Mean:{}'.format(round(epoch_loss_dict['loss_objectness'], 10), round(epoch_loss_dict['loss_rpn_box_reg'], 10) / len(dataset)))
    print('•Classification:\n\t•Cumulative: {}\n\t•Mean:{}'.format(round(epoch_loss_dict['loss_classifier'], 10), round(epoch_loss_dict['loss_rpn_box_reg'], 10) / len(dataset)))
    print('•Regression:\n\t•Cumulative: {}\n\t•Mean:{}'.format(round(epoch_loss_dict['loss_box_reg'], 10), round(epoch_loss_dict['loss_rpn_box_reg'], 10) / len(dataset)))

    print('-' * 80)
    print()


if __name__ == '__main__':
    # Defining hyperparameters for FasterRCNN
    hyperparameters = {
        'batch_size': 1,
        'num_workers': 5,
        'num_epochs': 1,
        'device': 'cpu',
        'dataset_type': SB_Detection.TRAIN,
        'network': {
            'pretrained': FasterRCNN,
            'weights_file_path': None
        }
    }
    if torch.cuda.is_available():
        hyperparameters['device'] = 'cuda'

    print('Commencing Evaluation')
    get_losses(hyperparameters)
    print('Evaluation Completed')
