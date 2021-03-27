import tensorflow as tf

# Convert the data to a TF dataset 
# from this: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch/ 
def to_tf_dataset(data, batch_size=500):
    tf_data = tf.data.Dataset.from_tensor_slices((data))
    tf_data = tf_data.shuffle(buffer_size=1024).batch(batch_size)
    return tf_data


def split_data(data,test_split = 0.1, validate_split = 0.1):
    """ Splits data into train, validation and test set 
    By default dataset is split into:
        Train    : 80%
        Validate : 10%
        Test     : 10%
    """
    # Split into train, validate and test
    N_test = int(test_split * data.shape[0])
    N_validate = int(validate_split * data.shape[0])+N_test
    data_test = data[:N_test]
    data_validate = data[N_test:N_validate]
    data_train = data[N_validate:]
    return data_train, data_validate, data_test
