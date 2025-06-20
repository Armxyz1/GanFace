class Config:
    image_size = 64
    batch_size = 16  # Lower batch size for CPU
    num_epochs = 100
    lr = 1e-4
    early_stopping_patience = 5
    train_dir = "dataset/train_images"
    val_dir = "dataset/val_images"
    save_model_path = "unet64_model.pt"
    checkpoint_path = "checkpoint.pth"
    device = "cuda"  # Explicitly use CPU