import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)  
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = torch.sigmoid(self.conv_out(x))
        return x

device_std = 'cuda' if torch.cuda.is_available() else 'cpu'

class DAE(L.LightningModule):
    def __init__(self, lr=0.001, noise_factor=0.5, device=device_std):
        super().__init__()
        self.lr = lr
        self.noise_factor = noise_factor
        self.criterion = nn.MSELoss()
        self.dae = ConvDenoiser()
        self.std_device = device

    def forward(self, x):
        return self.dae(x)

    def training_step(self, batch, batch_idx):
        images, _ = batch
        images = images.to(self.device,torch.float32)
        noisy_imgs = images + self.noise_factor * torch.randn(*images.shape).to(self.std_device)
        noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)
        outputs = self(noisy_imgs).to(self.std_device)
        loss = self.criterion(outputs, images)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        images = images.to(self.device,torch.float32)
        noisy_imgs = images + self.noise_factor * torch.randn(*images.shape).to(self.std_device)
        noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)
        outputs = self(noisy_imgs).to(self.std_device)
        loss = self.criterion(outputs, images)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
