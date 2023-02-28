import tensorflow as tf


def load_image(filename):
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


def save_image(filename, image):
    """Saves an image to a PNG file."""

    image = tf.clip_by_value(image, 0, 1)
    image = tf.round(image * 255)
    image = tf.cast(image, tf.uint8)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)
