# GanFace
GanFace is a project focused on synthesizing facial images of dark-skinned individuals for data augmentation in computer vision tasks. The model used in based on a Generative Adversarial Network (GAN) architecture.

## Installation

Get necessary libraries by running:

```bash
pip install -r requirements.txt
```

Get dataset by running:

```bash
bash dataset.sh
```

Get trained checkpoints using:

```bash
cd checkpoints
bash get_ckpt.sh
cd ..
```

## Training

To train the model run the following command:

```bash
python src/train.py
```

## Results
Result corresponding to best epoch (epoch_006) is shown below:
![Sample Output](./samples/epoch_006.png)
