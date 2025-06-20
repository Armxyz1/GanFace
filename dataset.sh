gdown 1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86
mv fairface-img-margin025-trainval.zip dataset.zip
mkdir -p dataset
unzip dataset.zip -d dataset
gdown 1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH -O dataset/train.csv
python src/preprocess.py
rm -rf dataset/train
rm -rf dataset/val
rm dataset.zip