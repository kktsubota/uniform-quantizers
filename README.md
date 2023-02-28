# Uniform-Quantizers
This repository provides the source code of our two papers:
- "Comprehensive Comparisons of Uniform Quantizers for Deep Image Compression" in ICIP 2021
- "Comprehensive Comparisons of Uniform Quantization in Deep Image Compression" in IEEE Access 2023.

## Dataset Preparation
Prepare the ImageNet, Kodak, and CLIC datasets.

We removed images whose shorter edge is smaller than 256 pixels for ImageNet.

Download the list of images that we used in our experiments by the following commands.

```
mkdir datasets
wget https://github.com/kktsubota/uniform-quantizers/releases/download/pre/ImageNet256.txt -O datasets/ImageNet256.txt
```

## Contact
Feel free to contact me if there is any question: tsubota (a) hal.t.u-tokyo.ac.jp

## License
This code is licensed under MIT (if not specified in the code).

Some part of this code are modified and copied open-source code. For the part, I describe the original license. Please let me know if there is a license issue with code redistribution. If so, I will remove the code and provide the instructions to reproduce the work.
