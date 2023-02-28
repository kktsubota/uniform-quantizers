# import logging
import os
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import hydra

from network import (
    analysis_transform,
    hyper_analysis,
    hyper_synthesis,
    entropy_parameter,
    synthesis_transform,
    RoundingEntropyBottleneck,
    RoundingGaussianConditional,
    quantize,
)
from lib.ops import get_heatmap3D, mask_with_heatmap, transpose_NCHW_to_NHWC, transpose_NHWC_to_NCHW
from nonuq import NonUQEntropyModel, Config
from utils import load_image, quantize_image



SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
# logger = logging.getLogger(__name__)


@hydra.main(config_name="config.yaml")
def main(cfg):
    cfg["DATASET"]["TRAIN_ROOT"] = os.path.expanduser(cfg["DATASET"]["TRAIN_ROOT"])
    # Create input data pipeline.
    with tf.device("/cpu:0"):
        with open(
            os.path.join(hydra.utils.get_original_cwd(), cfg["DATASET"]["TRAIN_PATH"])
        ) as f:
            lines = f.readlines()
        train_files = [
            os.path.join(cfg["DATASET"]["TRAIN_ROOT"], line.rstrip()) for line in lines
        ]
        train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
        train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
        train_dataset = train_dataset.map(
            load_image, num_parallel_calls=cfg["TRAIN"]["WORKERS"]
        )
        train_dataset = train_dataset.map(
            lambda x: tf.random_crop(
                x, (cfg["TRAIN"]["PATCH_SIZE"], cfg["TRAIN"]["PATCH_SIZE"], 3)
            )
        )
        train_dataset = train_dataset.batch(cfg["TRAIN"]["BATCH_SIZE"])
        train_dataset = train_dataset.prefetch(32)

    num_pixels = cfg["TRAIN"]["BATCH_SIZE"] * cfg["TRAIN"]["PATCH_SIZE"] ** 2

    # Transform and compress the image, then remove batch dimension.
    x = train_dataset.make_one_shot_iterator().get_next()

    step = tf.train.create_global_step()
    decaying_iter = tf.cast(step - cfg["TRAIN"]["TAU_DECAY_ITERATION"], tf.float32)
    # if decaying_iter < 0, tau should be 0.5.
    tau = tf.minimum(
        0.5, 0.5 * tf.exp(-cfg["TRAIN"]["TAU_DECAY_FACTOR"] * decaying_iter)
    )

    k_decay_iter = tf.cast(step - cfg["TRAIN"]["K_ITERATION"], tf.float32)
    # equivalent with cfg["TRAIN"]["K_INIT"]
    k = cfg["TRAIN"]["K_INIT"] + tf.maximum(
        0.0, (cfg["TRAIN"]["K_END"] - cfg["TRAIN"]["K_INIT"]) * k_decay_iter / (cfg["TRAIN"]["ITERATIONS"] - cfg["TRAIN"]["K_ITERATION"])
    )

    if cfg["TRAIN"]["QUANTIZE_ENT"] == "NonU-Q":
        pc_config = Config()
        entropy_bottleneck = NonUQEntropyModel(pc_config)
    
    else:
        entropy_bottleneck = RoundingEntropyBottleneck(
            approx=cfg["TRAIN"]["QUANTIZE_ENT"]
        )

    # forward
    y = analysis_transform(x, cfg["NUM_FILTERS"], cfg["SHALLOW"], cfg["HEATMAP"])

    if cfg["HEATMAP"]:
        y_nchw = transpose_NHWC_to_NCHW(y)
        heatmap3D = get_heatmap3D(y_nchw)
        y_masked = mask_with_heatmap(y_nchw, heatmap3D)
        y = transpose_NCHW_to_NHWC(y_masked)

    if cfg["PRIOR"] != "factorized":
        if cfg["PRIOR"] == "hyper":
            z = hyper_analysis(abs(y), cfg["NUM_FILTERS"], cfg["SHALLOW"])
        else:
            z = hyper_analysis(y, cfg["NUM_FILTERS"], cfg["SHALLOW"])
        if cfg["TRAIN"]["FIX_QUA"]:
            z_tilde, z_likelihoods = entropy_bottleneck(z, training=False)
        else:
            z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)
        
        if cfg["TRAIN"]["FIX_QUA"] or cfg["TRAIN"]["QUANTIZE_DEC"] == cfg["TRAIN"]["QUANTIZE_ENT"]:
            z_dec = z_tilde
        else:
            z_dec = quantize(z, method=cfg["TRAIN"]["QUANTIZE_DEC"], tau=tau, k=k)
        phi = hyper_synthesis(z_dec, cfg["NUM_FILTERS"], cfg["SHALLOW"])

    if cfg["PRIOR"] == "context":
        if cfg["TRAIN"]["FIX_QUA"]:
            y_tilde, y_likelihoods, y_means, y_variances, y_probs = entropy_parameter(
                phi, y, cfg["NUM_FILTERS"], training=False, n_gmm=cfg["N_GMM"],
            )
        else:
            y_tilde, y_likelihoods, y_means, y_variances, y_probs = entropy_parameter(
                phi,
                y,
                cfg["NUM_FILTERS"],
                activation=cfg["TRAIN"]["QUANTIZE_ENT"],
                activation_ha=cfg["TRAIN"]["QUANTIZE_HA"],
                training=True,
                n_gmm=cfg["N_GMM"],
                tau=tau,
                k=k,
            )
    elif cfg["PRIOR"] == "hyper":
        sigma = phi
        scale_table = np.exp(
            np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)
        )
        conditional_bottleneck = RoundingGaussianConditional(sigma, scale_table, approx=cfg["TRAIN"]["QUANTIZE_ENT"])
        if cfg["TRAIN"]["FIX_QUA"]:
            y_tilde, y_likelihoods = conditional_bottleneck(y, training=False)
        else:
            y_tilde, y_likelihoods = conditional_bottleneck(y, training=True)
    else:
        y_tilde, y_likelihoods = entropy_bottleneck(y, training=True)

    if (
        cfg["TRAIN"]["FIX_QUA"]
        or cfg["TRAIN"]["QUANTIZE_DEC"] == cfg["TRAIN"]["QUANTIZE_ENT"]
    ):
        y_dec = y_tilde
    else:
        y_dec = quantize(y, method=cfg["TRAIN"]["QUANTIZE_DEC"], tau=tau, k=k)
    x_tilde = synthesis_transform(y_dec, cfg["NUM_FILTERS"], cfg["SHALLOW"])

    # Total number of bits divided by number of pixels.
    if cfg["PRIOR"] == "factorized":
        if cfg["TRAIN"]["QUANTIZE_ENT"] != "NonU-Q":
            y_likelihoods = tf.log(y_likelihoods) / -np.log(2)
        H_real = tf.reduce_sum(y_likelihoods) / num_pixels
        if cfg["TRAIN"]["MASKLOSS"]:
            mask = transpose_NCHW_to_NHWC(heatmap3D)
            H_mask = tf.reduce_sum(y_likelihoods * mask) / num_pixels
            train_bpp = 0.5 * (H_mask + H_real)
        else:
            train_bpp = H_real
    else:
        train_bpp = (
            tf.reduce_sum(tf.log(y_likelihoods)) + tf.reduce_sum(tf.log(z_likelihoods))
        ) / (-np.log(2) * num_pixels)

    train_msssim = tf.reduce_mean(tf.image.ssim_multiscale(x_tilde * 255, x * 255, 255))
    if cfg["DISTORTION"] == "MSSSIM":
        train_distortion = 1.0 - train_msssim
    else:
        train_mse = tf.reduce_mean(tf.squared_difference(x * 255, x_tilde * 255))
        train_distortion = train_mse

    train_loss = cfg["LAMBDA"] * train_distortion + train_bpp

    if cfg["TRAIN"]["QUANTIZE_ENT"] == "NonU-Q":
        reg = tf.to_float(pc_config.regularization_factor_centers)
        centers_reg = tf.identity(reg * tf.nn.l2_loss(entropy_bottleneck.centers), name='l2_reg')
        train_loss += centers_reg

    # Minimize loss and auxiliary loss, and execute update op.
    # https://stackoverflow.com/questions/40443402/the-learning-rate-change-for-the-momentum-optimizer
    decay_steps = cfg["TRAIN"]["ITERATIONS"] - cfg["TRAIN"]["DECAYED_ITERATIONS"]
    adjusted_lr = tf.train.exponential_decay(
        cfg["TRAIN"]["MAIN_LR"],
        step,
        decay_steps,
        cfg["TRAIN"]["DECAY_FACTOR"],
        staircase=True,
    )
    main_optimizer = tf.train.AdamOptimizer(learning_rate=adjusted_lr)
    if cfg["TRAIN"]["FIX_QUA"]:
        var_list = list()
        if "ent" in cfg["TRAIN"]["MODULE_LIST"]:
            var_list += entropy_bottleneck.variables
        if "dec" in cfg["TRAIN"]["MODULE_LIST"]:
            var_list += tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="synthesis"
            )
        if "hdec" in cfg["TRAIN"]["MODULE_LIST"]:
            var_list += tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="hyper_synthesis"
            )
        if "cent" in cfg["TRAIN"]["MODULE_LIST"]:
            var_list += tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="entropy_parameter"
            )
        main_step = main_optimizer.minimize(
            train_loss, global_step=step, var_list=var_list
        )
    else:
        main_step = main_optimizer.minimize(train_loss, global_step=step)

    if cfg["TRAIN"]["QUANTIZE_ENT"] == "NonU-Q":
        train_op = tf.group(main_step)
    elif (not cfg["TRAIN"]["FIX_QUA"]) or "ent" in cfg["TRAIN"]["MODULE_LIST"]:
        aux_optimizer = tf.train.AdamOptimizer(learning_rate=adjusted_lr)
        aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])
        train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
    else:
        train_op = tf.group(main_step)
    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("bpp", train_bpp)
    tf.summary.scalar("lr", adjusted_lr)
    tf.summary.scalar("msssim", train_msssim)

    tf.summary.image("original", quantize_image(x))
    tf.summary.image("reconstruction", quantize_image(x_tilde))

    hooks = [
        tf.train.StopAtStepHook(last_step=cfg["TRAIN"]["ITERATIONS"]),
        tf.train.NanTensorHook(train_loss),
    ]

    checkpoint_dir: str = "checkpoint"
    if cfg["TRAIN"]["CHECKPOINT_DIR"]:
        shutil.copytree(
            os.path.join(
                hydra.utils.get_original_cwd(), cfg["TRAIN"]["CHECKPOINT_DIR"]
            ),
            checkpoint_dir,
        )
    with tf.train.MonitoredTrainingSession(
        hooks=hooks,
        checkpoint_dir=checkpoint_dir,
        save_checkpoint_secs=1800,
        save_summaries_secs=600,
    ) as sess:
        while not sess.should_stop():
            sess.run(train_op)


if __name__ == "__main__":
    main()
