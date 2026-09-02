import tensorflow as tf
from keras.utils import image_dataset_from_directory
from backend.model import ImageClassifier

def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train_ = image_dataset_from_directory(
    'C:/Users/kinse/Desktop/Image classifier/seg_train',
    labels='inferred',
    label_mode='int',          # was 'binary' — wrong for 6 classes, see note below
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    'C:/Users/kinse/Desktop/Image classifier/seg_test',
    labels='inferred',
    label_mode='int',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

ds_train = ds_train_.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)
ds_valid = ds_valid_.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)


classifier = ImageClassifier()
classifier.train(ds_train, ds_valid, epochs=50, save_path="models/image_classifier/weights.weights.h5")