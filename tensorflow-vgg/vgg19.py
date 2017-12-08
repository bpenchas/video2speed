import os
import tensorflow as tf

import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]

class BallerVgg:
    def __init__(self, vgg19_npy_path):
        # if vgg19_npy_path is None:
        #     path = inspect.getfile(Vgg19)
        #     path = os.path.abspath(os.path.join(path, os.pardir))
        #     path = os.path.join(path, "vgg19.npy")
        #     vgg19_npy_path = path
        #     print(vgg19_npy_path)

        self.var_dict = {}
        self.trainable = True

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, batch):
        vgg_vol = self.vgg_volume(batch) # batch_size x 28 x 28 x 256

        evens = tf.strided_slice(vgg_vol,(0,0,0,0), (batch.shape[0],28,28,256), (2,1,1,1))
        odds = tf.strided_slice(vgg_vol,(1,0,0,0), (batch.shape[0],28,28,256), (2,1,1,1))

        self.concat_vol = tf.concat([evens, odds], 3)

        # Make some conv layers, make some fc layers, that are variable/trainable.
        self.conv4_1 = self.conv_layer_trainable(self.concat_vol, 512, 512, "baller_conv4_1")
        self.conv4_2 = self.conv_layer_trainable(self.conv4_1, 512, 512, "baller_conv4_2")
        self.conv4_3 = self.conv_layer_trainable(self.conv4_2, 512, 512, "baller_conv4_3")
        self.conv4_4 = self.conv_layer_trainable(self.conv4_3, 512, 512, "baller_conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'baller_pool4')

        self.conv5_1 = self.conv_layer_trainable(self.pool4, 512, 512, "baller_conv5_1")
        self.conv5_2 = self.conv_layer_trainable(self.conv5_1, 512, 512, "baller_conv5_2")
        self.conv5_3 = self.conv_layer_trainable(self.conv5_2, 512, 512, "baller_conv5_3")
        self.conv5_4 = self.conv_layer_trainable(self.conv5_3, 512, 512, "baller_conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'baller_pool5')

        self.fc6 = self.fc_layer_trainable(self.pool5, 25088, 4096, "baller_fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        # if train_mode is not None:
        #     self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        # elif self.trainable:
        #     self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer_trainable(self.relu6, 4096, 4096, "baller_fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        # if train_mode is not None:
        #     self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        # elif self.trainable:
        #     self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer_trainable(self.relu7, 4096, 20, "baller_fc8")

        self.prob = tf.nn.softmax(self.fc8, name="baller_prob")

    def vgg_volume(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')
        # 28 x 28 x 256

        return self.pool3

        # self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        # self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        # self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        # self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        # self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        # self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        # self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        # self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        # self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        # self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        # self.fc6 = self.fc_layer(self.pool5, "fc6")
        # assert self.fc6.get_shape().as_list()[1:] == [4096]
        # self.relu6 = tf.nn.relu(self.fc6)

        # self.fc7 = self.fc_layer(self.relu6, "fc7")
        # self.relu7 = tf.nn.relu(self.fc7)

        # self.fc8 = self.fc_layer(self.relu7, "fc8")

        # self.prob = tf.nn.softmax(self.fc8, name="prob")

        # self.data_dict = None
        # print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu
    
    def conv_layer_trainable(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
    
    def fc_layer_trainable(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)
        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var(self, initial_value, name, idx, var_name):
        # TODO add new_dict lookup here for learned parameters
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print(var_name, var.get_shape().as_list())
        assert var.get_shape() == initial_value.get_shape()

        return var