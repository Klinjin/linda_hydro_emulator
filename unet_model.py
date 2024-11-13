""" Full assembly of the parts to form the complete network """
import sys
sys.path.append('/pscratch/sd/l/lindajin/Pytorch-UNet/unet')
from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear: bool = False, use_fourier_features: bool = False):
        super(UNet, self).__init__()
        self.use_fourier_features = use_fourier_features
        if use_fourier_features:
            self.fourier_features = FourierFeatures(
                first=-2.0,
                last=1,
                step=1,
            )
        if use_fourier_features:
            n_channels *= 1 + self.fourier_features.num_features

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        

    def forward(self, x): #params(6,)-->MLP score(N,)=weight+bias
        x = self.maybe_concat_fourier(x)    
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def maybe_concat_fourier(self, x):
        if self.use_fourier_features:
            return torch.cat([x, self.fourier_features(x)], dim=1)
        return x
    
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)