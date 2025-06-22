import os
import torch
from torchvision.utils import save_image

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename, gen, disc, opt_gen, opt_disc):
    checkpoint = torch.load(filename)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    disc.load_state_dict(checkpoint["disc_state_dict"])
    opt_gen.load_state_dict(checkpoint["opt_gen"])
    opt_disc.load_state_dict(checkpoint["opt_disc"])
    return checkpoint['epoch']

def save_samples(generator, epoch, fixed_noise, device, folder="samples"):
    os.makedirs(folder, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise.to(device)).detach().cpu()
        save_image(fake, os.path.join(folder, f"epoch_{epoch:03d}.png"), normalize=True)
    generator.train()

def get_fake_metric(fake_images):
    b, c, h, w = fake_images.shape
    flat = fake_images.view(b, -1)
    normalized = flat / flat.norm(dim=1, keepdim=True)
    cosine_sim = normalized @ normalized.T
    diversity_score = 1.0 - cosine_sim.mean().item()
    return diversity_score
