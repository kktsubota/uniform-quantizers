# Comprehensive Comparisons of Uniform Quantizers for Deep Image Compression
Official implementation of "Comprehensive Comparisons of Uniform Quantizers for Deep Image Compression" in ICIP 2021.

## Dataset
Prepare ImageNet and the Kodak dataset.

We removed images whose shorter edge is smaller than 256 pixels for ImageNet.

Download the list of images that we used in our experiments by the following commands.

```
mkdir datasets
wget https://github.com/kktsubota/uniform-quantizers/releases/download/pre/ImageNet256.txt -O datasets/ImageNet256.txt
```

## Environment
* CUDA==10.0
* CUDNN==7.6.0
* Python

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
python evaluate.py /path/to/Kodak/images/ --qua_ent AUN-Q --checkpoint_dir checkpoints/l0.01_aun_aun/
```

You can train other combinations of approximation methods by specifying `--qua_ent` and `--qua_dec`.

Please select from `{AUN-Q, STE-Q, U-Q, SGA-Q}` for each option.

## Citation
```
@inproceedings{tsubotaICIP21,
    title = {Comprehensive Comparisons of Uniform Quantizers for Deep Image Compression},
    author = {Tsubota, Koki and Aizawa, Kiyoharu},
    booktitle = {ICIP},
    year = {2021},
    pages={2089-2093}
}
```
