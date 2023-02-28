import argparse
import os

import tensorflow.compat.v1 as tf


def read_png(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255
    return image


def quantize_image(image):
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def write_png(filename, image):
    """Saves an image to a PNG file."""
    image = quantize_image(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)


def resize_with_keeping_aspect_ratio(image):
    assert len(image.shape) == 3
    
    # channel first
    if image.shape[0] == 3:
        h, w = image.shape[1], image.shape[2]
    
    # channel last
    elif image.shape[2] == 3:
        h, w = image.shape[0], image.shape[1]
    
    else:
        raise RuntimeError("Non-supported image shape")

    ratio = 256 / min(h, w)
    h_new = ratio * h
    w_new = ratio * w
    return tf.resize(image, [h_new, w_new])


def prepare_dataset(args):
    # args.train_root, args.train_file, args.preprocess_threads, args.patch_size, args.batch_size
    with open(args.train_file) as f:
        lines = f.readlines()
    train_files = [os.path.join(args.train_root, line.strip()) for line in lines]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(
        read_png, num_parallel_calls=args.preprocess_threads
    )
    train_dataset = train_dataset.map(
        lambda x: tf.image.random_crop(x, (args.patchsize, args.patchsize, 3))
    )
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(32)
    return train_dataset


def add_test_args(subparsers):
    # 'compress' subcommand.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a PNG file, compresses it, and writes a TFCI file.",
    )

    # 'decompress' subcommand.
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a TFCI file, reconstructs the image, and writes back "
        "a PNG file.",
    )

    # Arguments for both 'compress' and 'decompress'.
    for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
        cmd.add_argument("input_file", help="Input filename.")
        cmd.add_argument(
            "output_file",
            nargs="?",
            help="Output filename (optional). If not provided, appends '{}' to "
            "the input filename.".format(ext),
        )
