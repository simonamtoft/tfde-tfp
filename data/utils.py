import tensorflow as tf

# Convert the data to a TF dataset 
# from this: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch/ 
def to_tf_dataset(data, batch_size=500):
    tf_data = tf.data.Dataset.from_tensor_slices((data))
    tf_data = tf_data.shuffle(buffer_size=1024).batch(batch_size)
    return tf_data
