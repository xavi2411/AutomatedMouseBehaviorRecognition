import tensorflow as tf


def generate_frame_embedding(frame, feature_extractor):
    """
    Generate an embedding for the given frame and feature extractor
    """
    frame = tf.convert_to_tensor(frame)
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    return feature_extractor(frame)