import tensorflow as tf
import pandas as pd
import cv2
import os

# Function to create a TFRecord example
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _string_feature(value):
    """Returns a bytes_list from a string."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))


def create_tfrecord(Filepath, label):
    image = cv2.imread(Filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    feature = {
        'image': _bytes_feature(image),
        'label': _string_feature(label)  # Use _string_feature for the label
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Read the CSV file
csv_file_path = './image_df_full.csv'
image_df = pd.read_csv(csv_file_path)

# Create a TFRecord file
tfrecord_file = './data/images.tfrecord'
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for idx, row in image_df.iterrows():
        Filepath = row['Filepath']  # This should be updated with the correct column name
        label = row['label']  # This should be updated with the correct column name
        tf_example = create_tfrecord(Filepath, label)
        writer.write(tf_example.SerializeToString())
        
# Returning the path to the created TFRecord file
tfrecord_file_path = tfrecord_file
tfrecord_file_path