import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt 


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(768, 256, bilinear)
        self.up2 = Up(384, 128, bilinear)
        self.up3 = Up(192, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


def process_salient_image(test_image_path, results_folder):
    device = "cpu"
    model = UNet(n_channels=3, n_classes=1).to(device)
    # Load the model weights
    model_path = 'D:\\Cardd\\unet_model_8_final.pth' 
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    test_image = load_and_transform_image(test_image_path)
    predicted_mask = predict_mask(model, test_image, device)
    
    to_pil = transforms.ToPILImage()
    mask_image = to_pil(predicted_mask)

    output_path = os.path.join(results_folder+"\\"+"mask"+"\\", os.path.basename(test_image_path))
    mask_image.save(output_path)
    print("I ma file path", test_image_path)
    last_token = test_image_path.split("\\")[-1] 
    print("I am the last token", last_token)
    result_image_path = results_folder+ "\\mask_"+last_token
    print(result_image_path)
    return f"{result_image_path}"


def load_and_transform_image(image_path, size=(256, 256)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        
        transforms.ToTensor()
    ])
    return transform(image)

def predict_mask(model, image_tensor, device):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    predicted_mask = torch.sigmoid(output).squeeze(0).cpu()
    return predicted_mask
