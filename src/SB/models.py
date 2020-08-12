import torch
from torch import nn
from torchvision.models.detection import FasterRCNN as FRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator


class FasterRCNN:

    def __init__(self, pretrained=True):
        backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained)
        num_classes = 4 + 1

        anchor_generator = AnchorGenerator(sizes=(40, 60, 150, 200, 250), aspect_ratios=(0.7, 1.0, 1.3))
        self.model = FRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)

    def train(self):
        self.model.train()

    def to(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def get_state_dict(self):
        return self.model.state_dict()

    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def fit_batch(self, images, target):
        return self.model(images, target)

    def eval_batch(self, images):
        return self.model(images)


class UNET_D4(nn.Module):
    input_channels = 3
    convolutional_kernel_size = 3
    convolutional_padding = convolutional_kernel_size // 2
    convolutional_padding_mode = 'zeros'

    def __init__(self):
        super().__init__()

        self.encoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=UNET_D4.input_channels, out_channels=64, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        self.encoder_block_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
        )

        self.encoder_block_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.encoder_block_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2),
        )

        self.decoder_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
        )

        self.decoder_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
        )

        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
        )

        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=UNET_D4.input_channels, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode),
        )

    def forward(self, tensor):
        # Encoder
        encoder_block_1_output = self.encoder_block_1(tensor)
        encoder_block_2_output = self.encoder_block_2(encoder_block_1_output)
        encoder_block_3_output = self.encoder_block_3(encoder_block_2_output)
        encoder_block_4_output = self.encoder_block_4(encoder_block_3_output)

        # Bottleneck
        tensor = self.bottleneck(encoder_block_4_output)

        # Decoder
        tensor = self.decoder_block_4(torch.cat((tensor, encoder_block_4_output), dim=1))
        tensor = self.decoder_block_3(torch.cat((tensor, encoder_block_3_output), dim=1))
        tensor = self.decoder_block_2(torch.cat((tensor, encoder_block_2_output), dim=1))
        tensor = self.decoder_block_1(torch.cat((tensor, encoder_block_1_output), dim=1))

        return tensor
