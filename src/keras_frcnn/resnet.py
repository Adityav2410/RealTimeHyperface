# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''

from __future__ import print_function
from __future__ import absolute_import

from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed

import keras.layers.merge
from keras import backend as K

from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):

    # identity block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(input_tensor)
    shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):

    # conv block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# def nn_base(input_tensor=None, trainable=False):

#     # Determine proper input shape
#     if K.image_dim_ordering() == 'th':
#         input_shape = (3, None, None)
#     else:
#         input_shape = (None, None, 3)

#     if input_tensor is None:
#         img_input = Input(shape=input_shape)
#     else:
#         if not K.is_keras_tensor(input_tensor):
#             img_input = Input(tensor=input_tensor, shape=input_shape)
#         else:
#             img_input = input_tensor

#     if K.image_dim_ordering() == 'tf':
#         bn_axis = 3
#     else:
#         bn_axis = 1

#     x = ZeroPadding2D((3, 3))(img_input)

#     x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable = trainable)(x)
#     x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
#     x = Activation('relu')(x)
#     x = MaxPooling2D((3, 3), strides=(2, 2))(x)

#     x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable = trainable)
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable = trainable)
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable = trainable)

#     x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable = trainable)
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable = trainable)
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable = trainable)
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable = trainable)

#     x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable = trainable)
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable = trainable)
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable = trainable)
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable = trainable)
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable = trainable)
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable = trainable)

#     return x

# def vgg_b1(input_tensor=None, trainable=False):

#     # Determine proper input shape
#     if K.image_dim_ordering() == 'th':
#         input_shape = (3, None, None)
#     else:
#         input_shape = (None, None, 3)

#     if input_tensor is None:
#         img_input = Input(shape=input_shape)
#     else:
#         if not K.is_keras_tensor(input_tensor):
#             img_input = Input(tensor=input_tensor, shape=input_shape)
#         else:
#             img_input = input_tensor

#     if K.image_dim_ordering() == 'tf':
#         bn_axis = 3
#     else:
#         bn_axis = 1


# 	# Block 1
# 	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
# 	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
# 	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

#     return x


# def vgg_b2(input_tensor=None, trainable=False):
# 	# Block 2
# 	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(input_tensor)
# 	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
# 	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
# 	return x

# def vgg_b3(input_tensor=None, trainable=False):
# 	# Block 3
# 	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(input_tensor)
# 	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
# 	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
# 	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
# 	return x

# def vgg_b4(input_tensor=None, trainable=False):
# 	# Block 4
# 	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(input_tensor)
# 	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
# 	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
# 	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
# 	return x

# def vgg_b5(input_tensor=None, trainable=False):
# 	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
# 	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
# 	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
# 	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
# 	return x



def classifier_layers(x, input_shape, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    # (hence a smaller stride in the region that follows the ROI pool)
    # if K.backend() == 'tensorflow':
    #     x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
    # elif K.backend() == 'theano':
    #     x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(1, 1), trainable=trainable)

    # x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    # x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    # x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

    x = TimeDistributed(Conv2D(192, (1, 1), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Flatten())(x)
    x= TimeDistributed(Dense(3072, activation = 'relu'))(x)
    x_1 = TimeDistributed(Dense(512, activation='relu'))(x)
    x_2 = TimeDistributed(Dense(512, activation='relu'))(x)
    x_3 = TimeDistributed(Dense(512, activation='relu'))(x)
    x_4 = TimeDistributed(Dense(512, activation='relu'))(x)
    x_5 = TimeDistributed(Dense(512, activation='relu'))(x)

    x_1 = TimeDistributed(Dense(2, activation='softmax'), name = 'fc_detection')(x_1)
    x_2 = TimeDistributed(Dense(42, activation='linear'), name = 'fc_landmarks')(x_2)
    x_3 = TimeDistributed(Dense(21, activation='sigmoid'), name = 'fc_visibility')(x_3)
    x_4 = TimeDistributed(Dense(3, activation='linear'), name = 'fc_pose')(x_4)
    x_5 = TimeDistributed(Dense(2, activation='softmax'), name = 'fc_gender')(x_5)

    return(x1, x_2, x_3, x_4, x_5)


def rpn(base_layers,num_anchors):

    x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return([x_class, x_regr, base_layers])

# def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):

#     # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

#     if K.backend() == 'tensorflow':
#         pooling_regions = 14
#         input_shape = (num_rois,14,14,1024)
#     elif K.backend() == 'theano':
#         pooling_regions = 7
#         input_shape = (num_rois,1024,7,7)

#     out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
#     out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

#     out = TimeDistributed(Flatten())(out)

#     out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
#     # note: no regression target for bg class
#     out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
#     return [out_class, out_regr]

def classifier(vgg1, vgg3, vgg5, input_rois_1, input_rois_3, input_rois_5, num_rois, nb_classes = 21, trainable=False):
	"""
	* input_rois.shape = (C.num_rois, 4) := bbox coordinates in the shared_layer reference frame
	* num_rois = C.num_rois
	"""
	# compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
	# Convert each of the cropped windows from vgg1,vgg3,vgg5 to 6,6,256

	if K.backend() == 'tensorflow':
	    pooling_regions = 6
	    input_shape = (num_rois,6,6,768)
	elif K.backend() == 'theano':
	    pooling_regions = 6
	    input_shape = (num_rois,768,6,6)

	out_roi_pool_vgg1 = RoiPoolingConv(pooling_regions, num_rois)([vgg1, input_rois_1])
	out_roi_pool_vgg3 = RoiPoolingConv(pooling_regions, num_rois)([vgg3, input_rois_3])
	out_roi_pool_vgg5 = RoiPoolingConv(pooling_regions, num_rois)([vgg5, input_rois_5])

	# Make the numchannels 256 for all
	out_roi_pool_vgg1 = TimeDistributed(Convolution2D(256, (1, 1), activation='relu'))(out_roi_pool_vgg1)
	out_roi_pool_vgg3 = TimeDistributed(Convolution2D(256, (1, 1), activation='relu'))(out_roi_pool_vgg3)
	out_roi_pool_vgg5 = TimeDistributed(Convolution2D(256, (1, 1), activation='relu'))(out_roi_pool_vgg5)

	# Concatenate roi_pooled features
	out_roi_pool = keras.layers.merge.Concatenate([out_roi_pool_vgg1, out_roi_pool_vgg3, out_roi_pool_vgg5], axis = 3)


	x1, x_2, x_3, x_4, x_5 = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)


	#out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
	# note: no regression target for bg class
	#out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
	return [x1, x_2, x_3, x_4, x_5]
