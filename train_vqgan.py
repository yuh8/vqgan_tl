import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
import wandb
from lpips import LPIPS
import argparse

# Set environment variables for distributed training
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"


# Residual Block for the encoder and decoder
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        if not self.same_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)

        if not self.same_channels:
            x = self.shortcut(x)

        return h + x


# Attention Module
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = channels**-0.5

    def forward(self, x):
        residual = x
        n, c, h, w = x.shape

        # Normalize
        x = self.norm(x)

        # Compute q, k, v
        q = self.q(x).reshape(n, c, -1)
        k = self.k(x).reshape(n, c, -1)
        v = self.v(x).reshape(n, c, -1)

        # Compute attention
        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to v
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.reshape(n, c, h, w)
        out = self.proj_out(out)

        return out + residual


# Encoder architecture
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, embedding_dim, attn_resolutions=[16]):
        super().__init__()

        # Initial convolution
        modules = [
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, stride=1, padding=1)
        ]

        # Down blocks with residual blocks
        resolution = 32  # Assuming CIFAR-10 input (32x32)
        for i in range(len(hidden_dims) - 1):
            in_channels = hidden_dims[i]
            out_channels = hidden_dims[i + 1]

            # Residual block
            modules.append(ResidualBlock(in_channels, in_channels))

            # Add attention if at specified resolution
            if resolution in attn_resolutions:
                modules.append(AttentionBlock(in_channels))

            # Downsample
            modules.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
            resolution //= 2

            # Additional residual block after downsampling
            modules.append(ResidualBlock(out_channels, out_channels))

        # Final blocks
        modules.append(ResidualBlock(hidden_dims[-1], hidden_dims[-1]))
        modules.append(AttentionBlock(hidden_dims[-1]))
        modules.append(ResidualBlock(hidden_dims[-1], hidden_dims[-1]))

        # Output convolution
        modules.append(nn.GroupNorm(8, hidden_dims[-1]))
        modules.append(nn.SiLU())
        modules.append(
            nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=3, padding=1)
        )

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)


# Decoder architecture
class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dims, out_channels, attn_resolutions=[16]):
        super().__init__()
        hidden_dims = hidden_dims.copy()
        hidden_dims.reverse()

        # Initial convolution
        modules = [
            nn.Conv2d(embedding_dim, hidden_dims[0], kernel_size=3, stride=1, padding=1)
        ]

        # Initial blocks
        modules.append(ResidualBlock(hidden_dims[0], hidden_dims[0]))
        modules.append(AttentionBlock(hidden_dims[0]))
        modules.append(ResidualBlock(hidden_dims[0], hidden_dims[0]))

        # Up blocks with residual blocks
        resolution = 32 // (2 ** (len(hidden_dims) - 1))  # Starting resolution
        for i in range(len(hidden_dims) - 1):
            in_channels = hidden_dims[i]
            out_channels_block = hidden_dims[i + 1]

            # Residual block
            modules.append(ResidualBlock(in_channels, in_channels))

            # Add attention if at specified resolution
            if resolution in attn_resolutions:
                modules.append(AttentionBlock(in_channels))

            # Upsample
            modules.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels_block, kernel_size=4, stride=2, padding=1
                )
            )
            resolution *= 2

            # Additional residual block after upsampling
            modules.append(ResidualBlock(out_channels_block, out_channels_block))

        # Output blocks
        modules.append(nn.GroupNorm(8, hidden_dims[-1]))
        modules.append(nn.SiLU())
        modules.append(
            nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1)
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.decoder(x)


# Vector Quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.num_embeddings, 1.0 / self.num_embeddings
        )

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Convert quantized from BHWC -> BCHW
        return (
            quantized.permute(0, 3, 1, 2).contiguous(),
            loss,
            perplexity,
            encoding_indices,
        )


# Discriminator for VQGAN
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features_d=64, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()

        # Initial layer
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, features_d, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            )
        )

        # Middle layers
        for i in range(1, n_layers):
            in_features = features_d * (2 ** (i - 1))
            out_features = features_d * (2**i)
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_features, out_features, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(out_features),
                    nn.LeakyReLU(0.2),
                )
            )

        # Final layer
        in_features = features_d * (2 ** (n_layers - 1))
        self.layers.append(
            nn.Sequential(nn.Conv2d(in_features, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, x):
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)

        return feature_maps


# VQGAN Model
class VQGAN(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        hidden_dims=[128, 256, 512],
        embedding_dim=256,
        num_embeddings=1024,
        commitment_cost=0.25,
        disc_start_step=10000,
        disc_weight=0.8,
        perceptual_weight=1.0,
        learning_rate_g=1e-4,
        learning_rate_d=4e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(in_channels, hidden_dims, embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_dims, in_channels)
        self.discriminator = Discriminator(in_channels=in_channels)

        self.perceptual_loss = LPIPS(net="vgg").eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False

        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        self.disc_start_step = disc_start_step
        self.disc_weight = disc_weight
        self.perceptual_weight = perceptual_weight

        # For logging
        self.global_step_counter = 0

    def encode(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss, perplexity, _ = self.vq_layer(encoded)
        return quantized, vq_loss, perplexity

    def decode(self, quantized):
        return self.decoder(quantized)

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss, perplexity, _ = self.vq_layer(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, vq_loss, perplexity

    def _calculate_adaptive_weight(self, recon_loss, g_loss):
        # Adaptive weight calculation to balance reconstruction and adversarial loss
        recon_grads = torch.autograd.grad(
            recon_loss, self.decoder.decoder[-1].weight, retain_graph=True
        )[0]
        g_grads = torch.autograd.grad(
            g_loss, self.decoder.decoder[-1].weight, retain_graph=True
        )[0]

        recon_norm = torch.norm(recon_grads)
        g_norm = torch.norm(g_grads)

        if recon_norm > 0 and g_norm > 0:
            return 0.8 * (recon_norm / g_norm)
        else:
            return torch.tensor(0.8, device=recon_loss.device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        Args:
            batch (_type_): training data batch
            batch_idx (_type_): batch index
            optimizer_idx (_type_): which optimizer is used: generator or discriminator

        Returns:
            loss
        """
        real_images, _ = batch
        self.global_step_counter += 1

        # Generate reconstructions
        reconstructed, vq_loss, perplexity = self(real_images)

        # Calculate reconstruction loss - combination of L1 and perceptual loss
        recon_loss = F.l1_loss(reconstructed, real_images)
        p_loss = self.perceptual_loss(real_images, reconstructed).mean()

        # Combined reconstruction loss
        rec_loss = recon_loss + self.perceptual_weight * p_loss + vq_loss

        # Log reconstruction phase metrics
        self.log("train_recon_loss", recon_loss, sync_dist=True)
        self.log("train_perceptual_loss", p_loss, sync_dist=True)
        self.log("train_vq_loss", vq_loss, sync_dist=True)
        self.log("train_perplexity", perplexity, sync_dist=True)

        # Determine if discriminator training is active
        disc_training_active = self.global_step_counter >= self.disc_start_step

        # Generator step
        if optimizer_idx == 0:
            # Adversarial loss only if discriminator training is active
            if disc_training_active:
                # Discriminator output on reconstructed images
                disc_fake_outputs = self.discriminator(reconstructed)

                # Generator adversarial loss (fool the discriminator)
                # simply just maximizing logits
                g_loss = -torch.mean(disc_fake_outputs[-1])

                # Calculate adaptive weight to balance reconstruction and adversarial loss
                adaptive_weight = self._calculate_adaptive_weight(rec_loss, g_loss)

                # Combined generator loss
                g_total_loss = rec_loss + adaptive_weight * g_loss

                self.log("train_g_loss", g_loss, sync_dist=True)
                self.log("train_adaptive_weight", adaptive_weight, sync_dist=True)
            else:
                # Early training - focus only on reconstruction
                g_total_loss = rec_loss

            self.log("train_g_total_loss", g_total_loss, sync_dist=True)

            if batch_idx % 100 == 0 and self.trainer.is_global_zero:
                # Log images to W&B
                self._log_images(real_images, reconstructed, "train")

            return g_total_loss

        # Discriminator step
        elif optimizer_idx == 1 and disc_training_active:
            # Real images
            disc_real_outputs = self.discriminator(real_images)

            # Fake images - compute without gradients for efficiency
            with torch.no_grad():
                reconstructed, _, _ = self(real_images)

            disc_fake_outputs = self.discriminator(reconstructed.detach())

            # Hinge loss for discriminator
            d_loss_real = torch.mean(F.relu(1.0 - disc_real_outputs[-1]))
            d_loss_fake = torch.mean(F.relu(1.0 + disc_fake_outputs[-1]))
            d_loss = d_loss_real + d_loss_fake

            self.log("train_d_loss", d_loss, sync_dist=True)
            self.log("train_d_real", d_loss_real, sync_dist=True)
            self.log("train_d_fake", d_loss_fake, sync_dist=True)

            return d_loss

    def validation_step(self, batch, batch_idx):
        real_images, _ = batch
        reconstructed, vq_loss, perplexity = self(real_images)

        # Calculate losses
        recon_loss = F.l1_loss(reconstructed, real_images)
        p_loss = self.perceptual_loss(real_images, reconstructed).mean()

        # Log metrics
        self.log("val_recon_loss", recon_loss, sync_dist=True)
        self.log("val_perceptual_loss", p_loss, sync_dist=True)
        self.log("val_vq_loss", vq_loss, sync_dist=True)
        self.log("val_perplexity", perplexity, sync_dist=True)

        if batch_idx == 0 and self.trainer.is_global_zero:
            # Log images to W&B
            self._log_images(real_images, reconstructed, "val")

    def _log_images(self, input_images, reconstructions, stage):
        # Convert images from [-1, 1] to [0, 1] range
        input_images = (input_images + 1) / 2
        reconstructions = (reconstructions + 1) / 2

        # Create grid of images
        input_grid = torchvision.utils.make_grid(input_images[:8].clamp(0, 1))
        recon_grid = torchvision.utils.make_grid(reconstructions[:8].clamp(0, 1))

        # Log to W&B
        self.logger.experiment.log(
            {
                f"{stage}_input_images": wandb.Image(input_grid),
                f"{stage}_reconstructions": wandb.Image(recon_grid),
            }
        )

    def configure_optimizers(self):
        """
        Adding parameter set in each optimizer ensures the separation of gradient updates
        When generator is updated, discriminator parameters are frozen and vice versa
        """
        opt_g = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.vq_layer.parameters()),
            lr=self.learning_rate_g,
            betas=(0.5, 0.9),
        )

        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate_d, betas=(0.5, 0.9)
        )

        return [opt_g, opt_d], []


# Data Module
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=64, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def prepare_data(self):
        # Download data if needed
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Split datasets
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            train_size = int(len(cifar_full) * 0.9)
            val_size = len(cifar_full) - train_size
            self.cifar_train, self.cifar_val = random_split(
                cifar_full,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# Main training script
def main():
    parser = argparse.ArgumentParser(description="Train VQGAN model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--learning_rate_g", type=float, default=1e-4, help="Generator learning rate"
    )
    parser.add_argument(
        "--learning_rate_d",
        type=float,
        default=4e-4,
        help="Discriminator learning rate",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=256, help="Dimension of VQ embeddings"
    )
    parser.add_argument(
        "--num_embeddings", type=int, default=1024, help="Number of VQ embeddings"
    )
    parser.add_argument(
        "--disc_start_step",
        type=int,
        default=10000,
        help="Global step to start discriminator training",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="vqgan-distributed-training")

    # Initialize data module
    dm = CIFAR10DataModule(batch_size=args.batch_size, num_workers=args.num_workers)

    # Initialize model
    model = VQGAN(
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        learning_rate_g=args.learning_rate_g,
        learning_rate_d=args.learning_rate_d,
        disc_start_step=args.disc_start_step,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="vqgan-{epoch:02d}-{val_recon_loss:.4f}",
        save_top_k=3,
        monitor="val_recon_loss",
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Setup logger
    wandb_logger = WandbLogger(project="vqgan-distributed-training", log_model=True)

    # Setup trainer with DDP strategy
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=DDPStrategy(find_unused_parameters=True),  # Needed for GAN training
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        precision=16,  # Use mixed precision for faster training
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # Train the model
    trainer.fit(model, datamodule=dm)

    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    # To run on a cluster, execute this script with:
    # python -m torch.distributed.launch --nproc_per_node=4 train_vqgan.py
    main()
