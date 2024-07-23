import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os


from torchvision import transforms
import torchvision.transforms.functional as TF

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = self.conv1(x)
        # print(f"Conv1 output range: {x.min().item()} to {x.max().item()}")
        
        x = self.bn1(x)
        # print(f"BN1 output range: {x.min().item()} to {x.max().item()}")
        
        x = self.relu(x)
        # print(f"ReLU1 output range: {x.min().item()} to {x.max().item()}")
        
        x = self.conv2(x)
        # print(f"Conv2 output range: {x.min().item()} to {x.max().item()}")
        
        x = self.bn2(x)
        # print(f"BN2 output range: {x.min().item()} to {x.max().item()}")
        
        x = self.relu(x)
        # print(f"ReLU2 output range: {x.min().item()} to {x.max().item()}")
        
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
        for idx, encoder in enumerate(self.encoder_blocks):
            # print(f"Evaluating encoder block {idx}")
            x, features = encoder(x)
            if torch.isnan(x).any():
                print(f"NaN detected in encoder block {idx}")
            skip_connections.append(features)
        
        # print(f"Evaluating bottleneck")
        x = self.bottleneck(x)

        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse the list
        for idx, decoder in enumerate(self.decoder_blocks):
            # print(f"Evaluating decoder block {idx}")
            x = decoder(x, skip_connections[idx])

        # Final output layer
        return self.final_conv(x)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class UNetLightningModule(pl.LightningModule):
    def __init__(self, in_channels, out_channels, features, learning_rate):
        super().__init__()

        self.model = UNet(in_channels, out_channels, features)
        self.learning_rate = learning_rate

    # # initialise weights
    # def configure_model(self):
    #     self.apply(weights_init)

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

        # Calculate various metrics
        
        # Calculate count error
        true_count = y.sum(dim=(1,2,3))
        # print(true_count)
        pred_count = y_hat.sum(dim=(1,2,3))
        # print(pred_count)
        count_error = torch.abs(true_count - pred_count)

        # Log all metrics
        self.log('val_count_error', count_error.mean(), prog_bar=True, on_step=False, on_epoch=True)

        loss = F.mse_loss(y_hat, y)
        self.log('val_mse_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return {
                'optimizer': optimizer,
                # 'gradient_clip_val': 0.1,
                # 'gradient_clip_algorithm': 'norm'
            }

# class CornKernelDataset(Dataset):
#     def __init__(self, image_dir, density_map_dir, transform=None):
#         self.image_dir = image_dir
#         self.density_map_dir = density_map_dir
#         self.transform = transform
#         self.image_files = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
        
#         image_path = os.path.join(self.image_dir, img_name + '.jpg')
#         density_map_path = os.path.join(self.density_map_dir, f'{img_name}_class_0_density.npy')

#         # Load image
#         image = Image.open(image_path).convert('RGB')
#         # Load density map
#         density_map = np.load(density_map_path)

#         if self.transform:
#             image = self.transform(image)
#             density_map = torch.from_numpy(density_map).float().unsqueeze(0)

#         return image, density_map

class JointTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, density_map):
        if isinstance(self.transform, transforms.RandomCrop):
            i, j, h, w = self.transform.get_params(image, self.transform.size)
            image = TF.crop(image, i, j, h, w)
            density_map = TF.crop(density_map, i, j, h, w)
        else:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            density_map = self.transform(density_map)
        return image, density_map

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

        image = Image.open(image_path).convert('RGB')
        density_map = np.load(density_map_path)
        density_map = torch.from_numpy(density_map).float().unsqueeze(0)

        if self.transform:
            for t in self.transform.transforms:
                if isinstance(t, JointTransform):
                    image, density_map = t(image, density_map)
                else:
                    image = t(image)

        return image, density_map

# class CornKernelDataModule(pl.LightningDataModule):
#     def __init__(self, batch_size, num_workers, train_image_dir, train_density_map_dir, 
#                  val_image_dir, val_density_map_dir,crop_size=None):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.train_image_dir = train_image_dir
#         self.train_density_map_dir = train_density_map_dir
#         self.val_image_dir = val_image_dir
#         self.val_density_map_dir = val_density_map_dir
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225])  # ImageNet stats
#         ])

#     def setup(self, stage=None):
#         self.train_dataset = CornKernelDataset(
#             image_dir=self.train_image_dir,
#             density_map_dir=self.train_density_map_dir,
#             transform=self.transform
#         )
        
#         self.val_dataset = CornKernelDataset(
#             image_dir=self.val_image_dir,
#             density_map_dir=self.val_density_map_dir,
#             transform=self.transform
#         )

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, 
#                          shuffle=True, num_workers=self.num_workers)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, 
#                          num_workers=self.num_workers)
    
#     def predict_dataloader(self) -> torch.Any:
#         return super().predict_dataloader()

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
            JointTransform(transforms.RandomCrop(128)),
            # JointTransform(transforms.RandomHorizontalFlip()),
            transforms.ToTensor(),  # This will only be applied to the image
        ])

        self.train_transform = self.transform
        self.val_transform = self.transform

    def setup(self, stage=None):
        self.train_dataset = CornKernelDataset(
            image_dir=self.train_image_dir,
            density_map_dir=self.train_density_map_dir,
            transform=self.train_transform
        )
        
        self.val_dataset = CornKernelDataset(
            image_dir=self.val_image_dir,
            density_map_dir=self.val_density_map_dir,
            transform=self.val_transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers)

class DensityMapVisualizationCallback(Callback):
    def __init__(self, val_samples, num_samples=4):
        super().__init__()
        self.val_imgs, self.val_density_maps = val_samples
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Move validation samples to the same device as the model
        val_imgs = self.val_imgs[:self.num_samples].to(pl_module.device)
        val_density_maps = self.val_density_maps[:self.num_samples].to(pl_module.device)
        
        # Get predictions
        pl_module.eval()
        with torch.no_grad():
            preds = pl_module(val_imgs)
        pl_module.train()
        

        # print(f"Validation Predictions Shape: {preds.shape}")
        # print(f"Ground truth map shape {val_density_maps.shape} ")

        # Create a figure to display images, ground truth, and predictions
        fig, axes = plt.subplots(self.num_samples, 3, figsize=(15, 5*self.num_samples))
        for i in range(self.num_samples):
            # Display input image
            axes[i, 0].imshow(val_imgs[i].cpu().permute(1, 2, 0))
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis('off')
            
            # Display ground truth density map
            axes[i, 1].imshow(val_density_maps[i].cpu().squeeze(), cmap='jet')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # Display predicted density map
            axes[i, 2].imshow(preds[i].cpu().squeeze(), cmap='jet')
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Log the figure to TensorBoard
        trainer.logger.experiment.add_figure("Validation Predictions", fig, trainer.global_step)
        plt.close(fig)

if __name__ == "__main__":

    # Define hyperparameters
    hparams = {
        # Model hyperparameters
        'in_channels': 3,
        'out_channels': 1,
        'features': [64, 128, 256, 512],  # UNet feature sizes
        'learning_rate': 1e-4,

        # Data hyperparameters
        'batch_size': 8,
        'num_workers': 1,

        # Training hyperparameters
        'max_epochs': 10,
        
        # Paths
        'train_image_dir': '../datasets/corn_kernel_density/train/256x256/sigma-1.8',
        'train_density_map_dir': '../datasets/corn_kernel_density/train/256x256/sigma-1.8',
        'val_image_dir': '../datasets/corn_kernel_density/val/256x256/sigma-1.8',
        'val_density_map_dir': '../datasets/corn_kernel_density/val/256x256/sigma-1.8',
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

    # Set up the data module
    data_module.setup()

    val_dataloader = data_module.val_dataloader()
    val_samples = next(iter(val_dataloader))

    # train_dataloader = data_module.train_dataloader()
    # train_samples = next(iter(train_dataloader))

    # Create visualization callback
    visualization_callback = DensityMapVisualizationCallback(val_samples)

    # Create progress bar callback
    progress_bar = TQDMProgressBar(refresh_rate=20)

    logger = TensorBoardLogger("../logs/tb_logs", name="unet_vanilla")
    logger.log_hyperparams(hparams)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=hparams['max_epochs'],
        accelerator="gpu",
        callbacks=[progress_bar,visualization_callback],
        logger=logger
    )

    # # Create tuner
    tuner = Tuner(trainer)

    # Find optimal learning rate
    lr_finder = tuner.lr_find(model, datamodule=data_module)
    new_lr = lr_finder.suggestion()
    model.learning_rate = new_lr
    print(f"Suggested Learning Rate: {new_lr}")

    # # Find optimal batch size
    # batch_size_finder = tuner.scale_batch_size(model, datamodule=data_module, mode='power')
    # new_batch_size = data_module.batch_size
    # print(f"Suggested Batch Size: {new_batch_size}")

    # # Update hparams with new values
    # hparams['learning_rate'] = new_lr
    # hparams['batch_size'] = new_batch_size

    # Log updated hyperparameters
    logger.log_hyperparams(hparams)


    # Train the model
    trainer.fit(model, data_module)

    # predictions = trainer.predict(dataloaders=predict_dataloader)