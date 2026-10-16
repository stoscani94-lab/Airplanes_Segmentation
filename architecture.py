import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# ==========================================
# 1. CUSTOM BASELINE (Scritta da zero)
# ==========================================
class CustomFCN(nn.Module):

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        
        # ENCODER (Estrazione features e riduzione spaziale)
        self.enc1 = self._conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._conv_block(128, 256)
        
        # DECODER (Ricostruzione spaziale)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(64, 64)
        
        # LAYER FINALE (Mappa ai num_classes senza attivazione, esce in logits)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Input iniziale:      # (N, 3, H, W)  --> Es: (64, 3, 512, 512)
        
        # Discesa              # Dimensioni dopo applicazione di ogni layer:
        x = self.enc1(x)       # (N, 64, H, W)       --> Es: (64, 64, 512, 512)
        x = self.pool1(x)      # (N, 64, H/2, W/2)   --> Es: (64, 64, 256, 256)
        x = self.enc2(x)       # (N, 128, H/2, W/2)  --> Es: (64, 128, 256, 256)
        x = self.pool2(x)      # (N, 128, H/4, W/4)  --> Es: (64, 128, 128, 128)
        x = self.enc3(x)       # (N, 256, H/4, W/4)  --> Es: (64, 256, 128, 128)
        
        # Risalita
        x = self.up1(x)        # (N, 128, H/2, W/2)  --> Es: (64, 128, 256, 256)
        x = self.dec1(x)       # (N, 128, H/2, W/2)  --> Es: (64, 128, 256, 256)
        x = self.up2(x)        # (N, 64, H, W)       --> Es: (64, 64, 512, 512)
        x = self.dec2(x)       # (N, 64, H, W)       --> Es: (64, 64, 512, 512)
        
        # Output
        return self.final_conv(x) # (N, num_classes, H, W) --> Es: (64, 21, 512, 512)
        
def get_custom_baseline(num_classes):
    return CustomFCN(in_channels=3, num_classes=num_classes)

# ==========================================
# 2. U-NET STANDARD (Skip Connections)
# ==========================================
def get_unet(num_classes):
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    )

# ==========================================
# 3. DEEPLAB V3+ (Atrous Convolutions / Multi-scala)
# ==========================================
def get_deeplabv3plus(num_classes):
    return smp.DeepLabV3Plus(
        encoder_name="resnet50", # Usa un backbone un po' più profondo tipicamente
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    )

# ==========================================
# 4. TRANSFORMER BASED (Self-Attention)
# ==========================================
def get_transformer_seg(num_classes):
    # FPN (Feature Pyramid Network) abbinato a un Mix Vision Transformer (mit_b0)
    # È l'architettura alla base del famoso SegFormer
    return smp.FPN(
        encoder_name="mit_b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    )