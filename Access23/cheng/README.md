# Cheng20
Specify the path to ImageNet for `DATASET.TRAIN_ROOT` in `config.yaml`.

## Environment
```bash
pip install -r requirements.txt
```

## Comparison in Uniform Quantization
**Training**
*AUN-Q, U-Q, (AUN-Q, STE-Q), (AUN-Q, U-Q), (AUN-Q, DS-Q), (U-Q, STE-Q), (U-Q, DS-Q)*
```bash
# (TRAIN.QUANTIZE_ENT, TRAIN.QUANTIZE_DEC) should be set to
# (AUN-Q, AUN-Q), (U-Q, U-Q), (AUN-Q, STE-Q), (AUN-Q, U-Q), (AUN-Q, DS-Q), (U-Q, STE-Q), (U-Q, DS-Q)
python train.py LAMBDA=0.0016 NUM_FILTERS=128 TRAIN.QUANTIZE_ENT="AUN-Q" TRAIN.QUANTIZE_DEC="AUN-Q"
python train.py LAMBDA=0.003 NUM_FILTERS=128 TRAIN.QUANTIZE_ENT="AUN-Q" TRAIN.QUANTIZE_DEC="AUN-Q"
python train.py LAMBDA=0.0075 NUM_FILTERS=128 TRAIN.QUANTIZE_ENT="AUN-Q" TRAIN.QUANTIZE_DEC="AUN-Q"
python train.py LAMBDA=0.015 NUM_FILTERS=192 TRAIN.QUANTIZE_ENT="AUN-Q" TRAIN.QUANTIZE_DEC="AUN-Q"
python train.py LAMBDA=0.03 NUM_FILTERS=192 TRAIN.QUANTIZE_ENT="AUN-Q" TRAIN.QUANTIZE_DEC="AUN-Q"
python train.py LAMBDA=0.045 NUM_FILTERS=192 TRAIN.QUANTIZE_ENT="AUN-Q" TRAIN.QUANTIZE_DEC="AUN-Q"
```

*STH-Q* (an example when `LAMBDA=0.0016`)
```bash
# the output is saved in outputs/year-month-day/hour-minute-second
python train.py LAMBDA=0.0016 NUM_FILTERS=128 TRAIN.QUANTIZE_ENT="AUN-Q" TRAIN.QUANTIZE_DEC="AUN-Q" TRAIN.ITERATIONS=960000
python train.py LAMBDA=0.0016 NUM_FILTERS=128 TRAIN.QUANTIZE_ENT="AUN-Q" TRAIN.QUANTIZE_DEC="AUN-Q" TRAIN.CHECKPOINT_DIR=outputs/year-month-day/hour-minute-second TRAIN.FIX_QUA=true
```

**Evaluation**
```bash
# e.g., --result_dir outputs/2022-06-20/12-50-15
python evaluate.py kodak --result_dir /path/to/output
```

## Comparison between uniform and non-uniform quantization
### AUN-Q (Uniform)
**Training**
```bash
python train.py PRIOR=factorized LAMBDA=0.001 TRAIN.QUANTIZE_ENT="AUN-Q" TRAIN.QUANTIZE_DEC="AUN-Q" HEATMAP=true TRAIN.MASKLOSS=true
python train.py PRIOR=factorized LAMBDA=0.003 TRAIN.QUANTIZE_ENT="AUN-Q" TRAIN.QUANTIZE_DEC="AUN-Q" HEATMAP=true TRAIN.MASKLOSS=true
python train.py PRIOR=factorized LAMBDA=0.01 TRAIN.QUANTIZE_ENT="AUN-Q" TRAIN.QUANTIZE_DEC="AUN-Q" HEATMAP=true TRAIN.MASKLOSS=true
python train.py PRIOR=factorized LAMBDA=0.03 TRAIN.QUANTIZE_ENT="AUN-Q" TRAIN.QUANTIZE_DEC="AUN-Q" HEATMAP=true TRAIN.MASKLOSS=true NUM_FILTERS=192
```

**Evaluation**
```bash
# --num_filters 192 for LAMBDA=0.03
python encoder.py --dataset kodak --num_filters 128 --checkpoint_dir outputs/2022-11-22/10-33-44/checkpoint --heatmap --prior factorized --quantizer NonU-Q
```

### NU-Q (Non-Uniform)
**Training**
```bash
python train.py PRIOR=factorized LAMBDA=0.0003 TRAIN.QUANTIZE_ENT="NonU-Q" TRAIN.QUANTIZE_DEC="NonU-Q" HEATMAP=true TRAIN.MASKLOSS=true
python train.py PRIOR=factorized LAMBDA=0.001 TRAIN.QUANTIZE_ENT="NonU-Q" TRAIN.QUANTIZE_DEC="NonU-Q" HEATMAP=true TRAIN.MASKLOSS=true
python train.py PRIOR=factorized LAMBDA=0.003 TRAIN.QUANTIZE_ENT="NonU-Q" TRAIN.QUANTIZE_DEC="NonU-Q" HEATMAP=true TRAIN.MASKLOSS=true
python train.py PRIOR=factorized LAMBDA=0.01 TRAIN.QUANTIZE_ENT="NonU-Q" TRAIN.QUANTIZE_DEC="NonU-Q" HEATMAP=true TRAIN.MASKLOSS=true NUM_FILTERS=192
```

**Evaluation**
```bash
# --num_filters 192 for LAMBDA=0.01
python encoder.py --dataset kodak --num_filters 128 --checkpoint_dir outputs/path/to/checkpoint --heatmap --prior factorized --quantizer NonU-Q
```

## Note
Our code is based on [the code by Zhengxue Cheng](https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention).
