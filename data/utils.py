import tensorflow as tf

# Convert the data to a TF dataset
def to_tf_dataset(data, batch_size=500):
    return tf.data.Dataset.from_tensor_slices((data)).shuffle(10000).batch(batch_size)
