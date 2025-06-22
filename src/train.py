import torch
from tqdm import tqdm
from utils import save_checkpoint, save_samples, get_fake_metric

def train_fn(gen, disc, loader, opt_gen, opt_disc, criterion, fixed_noise, device, val_loader, args):
    best_gen_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(args.start_epoch, args.num_epochs):
        loop = tqdm(loader, leave=True)
        gen_losses = []

        for real in loop:
            real = real.to(device)
            noise = torch.randn(real.size(0), args.z_dim, 1, 1).to(device)
            fake = gen(noise)

            ### Train Discriminator ###
            disc_real = disc(real).view(-1)
            disc_fake = disc(fake.detach()).view(-1)
            loss_disc = criterion(disc_real, torch.ones_like(disc_real)) + \
                        criterion(disc_fake, torch.zeros_like(disc_fake))
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ### Train Generator ###
            output = disc(fake).view(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            gen_losses.append(loss_gen.item())

            loop.set_description(f"Epoch [{epoch}/{args.num_epochs}]")
            loop.set_postfix(loss_gen=loss_gen.item(), loss_disc=loss_disc.item())

        avg_gen_loss = sum(gen_losses) / len(gen_losses)

        ### Generate samples ###
        save_samples(gen, epoch, fixed_noise, device)

        ### Validation Metric for Logging Only ###
        with torch.no_grad():
            val_imgs = next(iter(val_loader)).to(device)
            fake = gen(torch.randn(val_imgs.size(0), args.z_dim, 1, 1).to(device))
            metric = get_fake_metric(fake)

        print(f"[VAL] Metric at epoch {epoch}: {metric:.4f}")
        print(f"[VAL] Average Generator Loss at epoch {epoch}: {avg_gen_loss:.4f}")

        # Early stopping and checkpointing based on average generator loss
        if avg_gen_loss < best_gen_loss:
            best_gen_loss = avg_gen_loss
            early_stop_counter = 0
            save_checkpoint({
                'epoch': epoch,
                'gen_state_dict': gen.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_disc': opt_disc.state_dict(),
            }, args.best_checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch} with loss {avg_gen_loss:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter > args.patience:
                print("Early stopping.")
                break

        # Save the model state at the end of each epoch
        save_checkpoint({
            'epoch': epoch,
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'opt_gen': opt_gen.state_dict(),
            'opt_disc': opt_disc.state_dict(),
        }, args.checkpoint_path)
        
        
