import torch
from monai.networks.nets import Unet


class BrainSegModel(torch.nn.Module):
    def __init__(self):
        super(BrainSegModel, self).__init__()
        self.UNet = Unet(
            dimensions=2,
            in_channels=1,
            out_channels=1,
            channels=[32, 64, 128, 256, 512, 1024],
            strides=[2, 2, 2, 2, 2],
            kernel_size=3,
            up_kernel_size=3,
        )

    def forward(self, x):
        x = self.UNet(x)
        return x
