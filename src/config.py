import torch
class Config:
    image_size = 64
    batch_size = 32  # Lower batch size for CPU
    num_epochs = 100
    lr = 1e-4
    early_stopping_patience = 3
    train_dir = "dataset/train_images"
    val_dir = "dataset/val_images"
    save_model_path = "unet64_model.pt"
    checkpoint_path = "checkpoint.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"