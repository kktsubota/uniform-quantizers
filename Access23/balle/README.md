# Ballé17 and Ballé18
## Environment
* CUDA==10.0
* CUDNN==7.6.0
* Python

```
pip install pipenv
pipenv install
```

Refer to `Pipfile` if you download packages manually.


## Comparison in Uniform Quantization
**Training**

*Quantization except STH-Q* (the below shows an example of (U-Q, DS-Q))
```bash
# Ballé17
python main.py --verbose --checkpoint_dir checkpoints/U-Q_DS-Q_0.001 --qua_ent U-Q train --lambda 0.001 --qua_dec DS-Q --seed 0
python main.py --verbose --checkpoint_dir checkpoints/U-Q_DS-Q_0.003 --qua_ent U-Q train --lambda 0.003 --qua_dec DS-Q --seed 0
python main.py --verbose --checkpoint_dir checkpoints/U-Q_DS-Q_0.01 --qua_ent U-Q train --lambda 0.01 --qua_dec DS-Q --seed 0
python main.py --verbose --checkpoint_dir checkpoints/U-Q_DS-Q_0.03 --qua_ent U-Q --num_filters 192 train --lambda 0.03 --qua_dec DS-Q --seed 0

# Ballé18
python main_hp.py --verbose --checkpoint_dir checkpoints-hp/U-Q_DS-Q_0.001 --qua_ent U-Q train --lambda 0.001 --qua_dec DS-Q --seed 0
python main_hp.py --verbose --checkpoint_dir checkpoints-hp/U-Q_DS-Q_0.003 --qua_ent U-Q train --lambda 0.003 --qua_dec DS-Q --seed 0
python main_hp.py --verbose --checkpoint_dir checkpoints-hp/U-Q_DS-Q_0.01 --qua_ent U-Q train --lambda 0.01 --qua_dec DS-Q --seed 0
python main_hp.py --verbose --checkpoint_dir checkpoints-hp/U-Q_DS-Q_0.03 --qua_ent U-Q --num_filters 192 train --lambda 0.03 --qua_dec DS-Q --seed 0
```

*STH-Q* (an example when `lambda=0.001`)
```bash
# Ballé17
python main.py --verbose --checkpoint_dir checkpoints/STH-Q_0.001 --qua_ent AUN-Q train --lambda 0.001 --qua_dec AUN-Q --seed 0 --last_step 960000
python main.py --verbose --checkpoint_dir checkpoints/STH-Q_0.001 --qua_ent AUN-Q train --lambda 0.001 --qua_dec AUN-Q --seed 0 --fix_qua

# Ballé18
python main_hp.py --verbose --checkpoint_dir checkpoints-hp/STH-Q_0.001 --qua_ent AUN-Q train --lambda 0.001 --qua_dec AUN-Q --seed 0 --last_step 960000
python main_hp.py --verbose --checkpoint_dir checkpoints-hp/STH-Q_0.001 --qua_ent AUN-Q train --lambda 0.001 --qua_dec AUN-Q --seed 0 --fix_qua
```

**Evaluation**
```bash
# Ballé17
# kodak
python evaluate.py /path/to/kodak --model factorized --out kodak.csv --lambda 0.001 --num_filters 128 --checkpoint_dir checkpoints/U-Q_DS-Q_0.001
# clic
python evaluate.py /path/to/clic --model factorized --out clic.csv --lambda 0.001 --num_filters 128 --checkpoint_dir checkpoints/U-Q_DS-Q_0.001

# Ballé18
# kodak
python evaluate.py /path/to/kodak --model hyper --out kodak.csv --lambda 0.001 --num_filters 128 --checkpoint_dir checkpoints-hp/U-Q_DS-Q_0.001
# clic
python evaluate.py /path/to/clic --model hyper --out clic.csv --lambda 0.001 --num_filters 128 --checkpoint_dir checkpoints-hp/U-Q_DS-Q_0.001
```

## Comparison between uniform and non-uniform quantization
### AUN-Q (Uniform)
**Training**
```bash
# Ballé17
python main.py --verbose --autoencoder balle17 --qua_ent AUN-Q --checkpoint_dir checkpoints-add/B17hm_AUN-Q_0.001 --heatmap --num_filters 128 train --lambda 0.001 --qua_dec AUN-Q --seed 0 --distortion mse --mask-loss
python main.py --verbose --autoencoder balle17 --qua_ent AUN-Q --checkpoint_dir checkpoints-add/B17hm_AUN-Q_0.003 --heatmap --num_filters 128 train --lambda 0.003 --qua_dec AUN-Q --seed 0 --distortion mse --mask-loss
python main.py --verbose --autoencoder balle17 --qua_ent AUN-Q --checkpoint_dir checkpoints-add/B17hm_AUN-Q_0.01 --heatmap --num_filters 128 train --lambda 0.01 --qua_dec AUN-Q --seed 0 --distortion mse --mask-loss
python main.py --verbose --autoencoder balle17 --qua_ent AUN-Q --checkpoint_dir checkpoints-add/B17hm_AUN-Q_0.03 --heatmap --num_filters 192 train --lambda 0.03 --qua_dec AUN-Q --seed 0 --distortion mse --mask-loss

# Ballé18
python main.py --verbose --autoencoder balle18 --qua_ent AUN-Q --checkpoint_dir checkpoints-add/B18hm_AUN-Q_0.001 --heatmap --num_filters 128 train --lambda 0.001 --qua_dec AUN-Q --seed 0 --distortion mse --mask-loss
python main.py --verbose --autoencoder balle18 --qua_ent AUN-Q --checkpoint_dir checkpoints-add/B18hm_AUN-Q_0.003 --heatmap --num_filters 128 train --lambda 0.003 --qua_dec AUN-Q --seed 0 --distortion mse --mask-loss
python main.py --verbose --autoencoder balle18 --qua_ent AUN-Q --checkpoint_dir checkpoints-add/B18hm_AUN-Q_0.01 --heatmap --num_filters 128 train --lambda 0.01 --qua_dec AUN-Q --seed 0 --distortion mse --mask-loss
python main.py --verbose --autoencoder balle18 --qua_ent AUN-Q --checkpoint_dir checkpoints-add/B18hm_AUN-Q_0.03 --heatmap --num_filters 192 train --lambda 0.03 --qua_dec AUN-Q --seed 0 --distortion mse --mask-loss
```

**Evaluation**
```bash
# Ballé17
python evaluate.py /path/to/kodak --model factorized --autoencoder balle17 --lambda 0.001 --distortion mse --checkpoint_dir checkpoints-add/B17hm_AUN-Q_0.001/ --out kodak.csv --qua_ent AUN-Q --heatmap

# Ballé18
python evaluate.py /path/to/kodak --model factorized --autoencoder balle18 --lambda 0.001 --distortion mse --checkpoint_dir checkpoints-add/B18hm_AUN-Q_0.001/ --out kodak.csv --qua_ent AUN-Q --heatmap
```

### NU-Q (Non-Uniform)
**Training**
```bash
# Ballé17
python main.py --verbose --autoencoder balle17 --qua_ent NonU-Q --checkpoint_dir checkpoints-add/B17hm_NonU-Q_0.0003 --heatmap --num_filters 128 train --lambda 0.0003 --qua_dec NonU-Q --seed 0 --distortion mse --regularization_factor_centers 0.0 --stop_gradient --mask-loss
python main.py --verbose --autoencoder balle17 --qua_ent NonU-Q --checkpoint_dir checkpoints-add/B17hm_NonU-Q_0.001 --heatmap --num_filters 128 train --lambda 0.001 --qua_dec NonU-Q --seed 0 --distortion mse --regularization_factor_centers 0.0 --stop_gradient --mask-loss
python main.py --verbose --autoencoder balle17 --qua_ent NonU-Q --checkpoint_dir checkpoints-add/B17hm_NonU-Q_0.003 --heatmap --num_filters 128 train --lambda 0.003 --qua_dec NonU-Q --seed 0 --distortion mse --regularization_factor_centers 0.0 --stop_gradient --mask-loss
python main.py --verbose --autoencoder balle17 --qua_ent NonU-Q --checkpoint_dir checkpoints-add/B17hm_NonU-Q_0.01 --heatmap --num_filters 192 train --lambda 0.01 --qua_dec NonU-Q --seed 0 --distortion mse --regularization_factor_centers 0.0 --stop_gradient --mask-loss

# Ballé18
python main.py --verbose --autoencoder balle18 --qua_ent NonU-Q --checkpoint_dir checkpoints-add/B18hm_NonU-Q_0.0003 --heatmap --num_filters 128 train --lambda 0.0003 --qua_dec NonU-Q --seed 0 --distortion mse --regularization_factor_centers 0.0 --stop_gradient --mask-loss
python main.py --verbose --autoencoder balle18 --qua_ent NonU-Q --checkpoint_dir checkpoints-add/B18hm_NonU-Q_0.001 --heatmap --num_filters 128 train --lambda 0.001 --qua_dec NonU-Q --seed 0 --distortion mse --regularization_factor_centers 0.0 --stop_gradient --mask-loss
python main.py --verbose --autoencoder balle18 --qua_ent NonU-Q --checkpoint_dir checkpoints-add/B18hm_NonU-Q_0.003 --heatmap --num_filters 128 train --lambda 0.003 --qua_dec NonU-Q --seed 0 --distortion mse --regularization_factor_centers 0.0 --stop_gradient --mask-loss
python main.py --verbose --autoencoder balle18 --qua_ent NonU-Q --checkpoint_dir checkpoints-add/B18hm_NonU-Q_0.01 --heatmap --num_filters 192 train --lambda 0.01 --qua_dec NonU-Q --seed 0 --distortion mse --regularization_factor_centers 0.0 --stop_gradient --mask-loss
```

**Evaluation**
```bash
# Ballé17
python evaluate.py /home/tsubota/data/compression/Kodak/images/ --model factorized --autoencoder balle17 --lambda 0.0003 --distortion mse --checkpoint_dir checkpoints-add/B17hm_NonU-Q_0.0003/ --out kodak.csv --qua_ent NonU-Q --heatmap

# Ballé18
python evaluate.py /home/tsubota/data/compression/Kodak/images/ --model factorized --autoencoder balle18 --lambda 0.0003 --distortion mse --checkpoint_dir checkpoints-add/B18hm_NonU-Q_0.0003/ --out kodak.csv --qua_ent NonU-Q --heatmap
```

## Note
Our code is based on the example codes in [Tensorflow Compression](https://github.com/tensorflow/compression/tree/v1.3/examples) licensed under Apache-2.0 (Copyright 2018 Google LLC).
