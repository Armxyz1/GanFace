import os
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from dataset import ImageDataset
from model import get_model, get_scheduler

def save_checkpoint(path, model, optimizer, epoch, best_val_loss, accelerator):
    if accelerator.is_main_process:
        torch.save({
            'model_state': accelerator.get_state_dict(model),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
        }, path)

def load_checkpoint(path, model, optimizer, accelerator):
    if not os.path.exists(path):
        return model, optimizer, 0, float("inf")  # No checkpoint, start fresh
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Resumed from checkpoint at epoch {checkpoint['epoch'] + 1}")
    return model, optimizer, checkpoint["epoch"] + 1, checkpoint["best_val_loss"]

def train(config):
    accelerator = Accelerator(cpu=True)

    model = get_model(config.image_size)
    noise_scheduler = get_scheduler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    train_data = ImageDataset(config.train_dir, config.image_size)
    val_data = ImageDataset(config.val_dir, config.image_size)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Load checkpoint if available
    model, optimizer, start_epoch, best_val_loss = load_checkpoint(
        config.checkpoint_path, model, optimizer, accelerator
    )
    patience_counter = 0

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
            noise = torch.randn_like(batch)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch.size(0),),
                device=batch.device
            ).long()

            noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy_images, timesteps).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}"):
                noise = torch.randn_like(batch)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (batch.size(0),),
                    device=batch.device
                ).long()
                noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)
                noise_pred = model(noisy_images, timesteps).sample
                val_loss += torch.nn.functional.mse_loss(noise_pred, noise).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        save_checkpoint(config.checkpoint_path, model, optimizer, epoch, best_val_loss, accelerator)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if accelerator.is_main_process:
                torch.save(model.state_dict(), config.save_model_path)
                print(f"Model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print("Early stopping triggered.")
                break

    print("Training finished.")
