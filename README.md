# uniform-quantizers
Official implementation of "Comprehensive Comparisons of Uniform Quantizers for Deep Image Compression".

## Dataset
Prepare ImageNet and the Kodak dataset.
We removed images whose shorter edge is smaller than 256 pixels for ImageNet.
You can download the list of images that we used in our experiments by the following commands.

```
mkdir datasets
wget https://github.com/fujibo/uniform-quantizers/releases/download/pre/ImageNet256.txt -O datasets/ImageNet256.txt
```


## Environment

```
pip install pipenv
pipenv install
```

Refer to `Pipfile` if you download packages manually.

## Usage

```bash
# train a model
python main.py --verbose --checkpoint_dir checkpoints/l0.01_aun_aun --qua_ent AUN-Q train --lambda 0.01 --qua_dec AUN-Q --train_root /path/to/ImageNet/train/

# evaluate the model
python evaluate.py /home/tsubota/data/compression/Kodak/images/ --qua_ent AUN-Q --checkpoint_dir checkpoints/l0.01_aun_aun/
```

You can train other combinations of approximation methods by specifying `--qua_ent` and `--qua_dec`.

## Citation
```
@inproceedings{tsubota,
    title = {Comprehensive Comparisons of Uniform Quantizers for Deep Image Compression},
    author = {Tsubota, Koki and Aizawa, Kiyoharu},
    booktitle = {ICIP},
    year = {2021}
}
```
