import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from dataset import FaceDataset
from train import train_fn
from utils import load_checkpoint

class Args:
    train_dir = "dataset/train_images"
    val_dir = "dataset/val_images"
    batch_size = 64
    z_dim = 100
    lr = 2e-4
    num_epochs = 200
    patience = 5
    image_size = 64
    checkpoint_path = "checkpoints/checkpoint.pth.tar"
    best_checkpoint_path = "checkpoints/best_checkpoint.pth.tar"
    start_epoch = 0

def main():
    args = Args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = FaceDataset(args.train_dir, args.image_size)
    val_dataset = FaceDataset(args.val_dir, args.image_size)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    gen = Generator(args.z_dim).to(device)
    disc = Discriminator().to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, args.z_dim, 1, 1).to(device)

    if os.path.exists(args.checkpoint_path):
        args.start_epoch = load_checkpoint(args.checkpoint_path, gen, disc, opt_gen, opt_disc) + 1

    train_fn(gen, disc, loader, opt_gen, opt_disc, criterion, fixed_noise, device, val_loader, args)

if __name__ == "__main__":
    main()
