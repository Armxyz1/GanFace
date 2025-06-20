from diffusers import UNet2DModel, DDPMScheduler

def get_model(image_size=64):
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512),
        down_block_types=("DownBlock2D",) * 5,
        up_block_types=("UpBlock2D",) * 5,
    )
    return model

def get_scheduler():
    return DDPMScheduler(num_train_timesteps=1000)
