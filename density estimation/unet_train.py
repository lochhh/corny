import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

import torch
from torch import nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=(3,3),stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self,x):
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)  # *2 because of concatenation

    def forward(self, x, skip_connection):
        upsampled = self.up(x)
        combined_features = torch.cat([upsampled, skip_connection], dim=1)
        return self.conv(combined_features)
        
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Encoder blocks
        for feature in features:
            self.encoder_blocks.append(EncoderBlock(in_channels, feature))
            in_channels = feature
        
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # Decoder blocks
        for feature in reversed(features):
            self.decoder_blocks.append(DecoderBlock(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for encoder in self.encoder_blocks:
            x, features = encoder(x)
            skip_connections.append(features)
        
        x = self.bottleneck(x)

        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse the list
        for idx, decoder in enumerate(self.decoder_blocks):
            x = decoder(x, skip_connections[idx])

        # Final output layer
        return self.final_conv(x)

class UNetLightningModule(pl.LightningModule):
    def __init__(self, in_channels, out_channels, features, learning_rate):
        super().__init__()
        self.model = UNet(in_channels, out_channels, features)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class CornKernelDataset(Dataset):
    def __init__(self, image_dir, density_map_dir, transform=None):
        self.image_dir = image_dir
        self.density_map_dir = density_map_dir
        self.transform = transform
        self.image_files = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        image_path = os.path.join(self.image_dir, img_name + '.jpg')
        density_map_path = os.path.join(self.density_map_dir, f'{img_name}_class_0_density.npy')

        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Load density map
        density_map = np.load(density_map_path)

        if self.transform:
            image = self.transform(image)
            density_map = torch.from_numpy(density_map).float().unsqueeze(0)

        return image, density_map

class CornKernelDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, train_image_dir, train_density_map_dir, 
                 val_image_dir, val_density_map_dir):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_image_dir = train_image_dir
        self.train_density_map_dir = train_density_map_dir
        self.val_image_dir = val_image_dir
        self.val_density_map_dir = val_density_map_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        self.train_dataset = CornKernelDataset(
            image_dir=self.train_image_dir,
            density_map_dir=self.train_density_map_dir,
            transform=self.transform
        )
        
        self.val_dataset = CornKernelDataset(
            image_dir=self.val_image_dir,
            density_map_dir=self.val_density_map_dir,
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers)

if __name__ == "__main__":

    # Define hyperparameters
    hparams = {
        # Model hyperparameters
        'in_channels': 3,
        'out_channels': 1,
        'features': [64, 128, 256, 512],  # UNet feature sizes
        'learning_rate': 1e-3,

        # Data hyperparameters
        'batch_size': 3,
        'num_workers': 1,

        # Training hyperparameters
        'max_epochs': 2,
        
        # Paths
        'train_image_dir': '../datasets/corn_yolo_no_segment/images/corn_kernel_train/resized',
        'train_density_map_dir': './maps/kernel-train',
        'val_image_dir': '../datasets/corn_yolo_no_segment/images/corn_kernel_val/resized',
        'val_density_map_dir': './maps/kernel-val',
    }

    
     # Create model
    model = UNetLightningModule(
        in_channels=hparams['in_channels'],
        out_channels=hparams['out_channels'],
        features=hparams['features'],
        learning_rate=hparams['learning_rate']
    )

    # Create data module
    data_module = CornKernelDataModule(
        batch_size=hparams['batch_size'],
        num_workers=hparams['num_workers'],
        train_image_dir=hparams['train_image_dir'],
        train_density_map_dir=hparams['train_density_map_dir'],
        val_image_dir=hparams['val_image_dir'],
        val_density_map_dir=hparams['val_density_map_dir']
    )

    # Create progress bar callback
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=hparams['max_epochs'],
        accelerator="gpu",
        callbacks=[progress_bar]
    )

    # Train the model
    trainer.fit(model, data_module)