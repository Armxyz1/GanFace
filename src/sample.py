import torch
from torchvision.utils import save_image
from .model import Generator

def generate_samples(checkpoint_path, num_samples=64, z_dim=100, out_file="generated.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = Generator(z_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen.eval()

    noise = torch.randn(num_samples, z_dim, 1, 1).to(device)
    with torch.no_grad():
        fake = gen(noise)
        save_image(fake, out_file, normalize=True)

if __name__ == "__main__":
    generate_samples("checkpoint.pth.tar")
