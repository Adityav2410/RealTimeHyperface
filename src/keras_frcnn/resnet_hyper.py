# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''




from __future__ import print_function
from __future__ import absolute_import

import pdb

from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, merge, \
    AveragePooling2D, TimeDistributed, Concatenate, concatenate,Lambda,Masking

from keras.layers.merge import Concatenate
from keras import backend as K

from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization
from pdb import set_trace as bp


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

def classifier_layers(x, trainable=False):

    x= TimeDistributed(Dense(3072, activation = 'relu'))(x)
    x_1 = TimeDistributed(Dense(512, activation='relu'),name = 'x_1')(x)
    x_2 = TimeDistributed(Dense(512, activation='relu'), name = 'x_2')(x)
    x_3 = TimeDistributed(Dense(512, activation='relu'), name = 'x_3')(x)
    x_4 = TimeDistributed(Dense(512, activation='relu'), name = 'x_4')(x)
    x_5 = TimeDistributed(Dense(512, activation='relu'), name = 'x_5')(x)
    x_6 = TimeDistributed(Dense(512, activation='relu'), name = 'x_6')(x)
    face_pred = TimeDistributed(Dense(2, activation='softmax'), name = 'fc_detection')(x_1)
    pose_pred = TimeDistributed(Dense(3, activation='linear'), name = 'fc_pose')(x_2)
    gender_pred = TimeDistributed(Dense(2, activation='softmax'), name = 'fc_gender')(x_3)
    viz_pred = TimeDistributed(Dense(21, activation='linear'), name = 'fc_vizibility')(x_4)
    landmark_pred = TimeDistributed(Dense(42, activation='linear'), name = 'fc_landmarks')(x_5)
    regr_pred = TimeDistributed(Dense(4, activation='linear'), name = 'fc_regr')(x_6)
    return([face_pred, pose_pred, gender_pred, viz_pred, landmark_pred, regr_pred])



def rpn( b1_feat, b2_feat, b4_feat ,num_anchors):

    x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(b4_feat)

    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return([x_class, x_regr, b1_feat, b2_feat, b4_feat ] )



def classifier(vgg1, vgg2, vgg4, input_rois_4, num_rois, nb_classes = 21, trainable=False):
# def classifier(vgg1, input_rois_4, num_rois, nb_classes = 21, trainable=False):
    """
    * input_rois.shape = (C.num_rois, 4) := bbox coordinates in the shared_layer reference frame
    * num_rois = C.num_rois
    """
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    # Convert each of the cropped windows from vgg1,vgg3,vgg5 to 6,6,256
    if K.backend() == 'tensorflow':
        pooling_regions_1 = 96
        pooling_regions_2 = 24
        pooling_regions_4 = 6


    input_rois_2 = Lambda( lambda x: 4*x)(input_rois_4)
    input_rois_1 = Lambda( lambda x: 4*x)(input_rois_2)


    out_roi_pool_vgg1 = RoiPoolingConv(pooling_regions_1, num_rois,name='roi_pool_1')([vgg1, input_rois_1])
    out_roi_pool_vgg2 = RoiPoolingConv(pooling_regions_2, num_rois,name='roi_pool2')([vgg2, input_rois_2])
    out_roi_pool_vgg4 = RoiPoolingConv(pooling_regions_4, num_rois,name='roi_pool_4')([vgg4, input_rois_4])


    # # Make the numchannels 256 for all
    out_roi_pool_vgg1 = TimeDistributed(Convolution2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu'), name= 'feat1_td_conv1')(out_roi_pool_vgg1)
    out_roi_pool_vgg1 = TimeDistributed(Convolution2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu'), name= 'feat1_td_conv2')(out_roi_pool_vgg1)
    # out_roi_pool_vgg1 = TimeDistributed(MaxPooling2D(256, strides = (2,2), padding='same'))(out_roi_pool_vgg1)
    out_roi_pool_vgg1 = TimeDistributed(Convolution2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu'), name= 'feat1_td_conv3')(out_roi_pool_vgg1)
    out_roi_pool_vgg1 = TimeDistributed(Convolution2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu'), name= 'feat1_td_conv4')(out_roi_pool_vgg1)


    out_roi_pool_vgg2 = TimeDistributed(Convolution2D(256, (3, 3), strides = (2,2), padding='same', activation='relu'), name= 'feat2_td_conv1')(out_roi_pool_vgg2)
    out_roi_pool_vgg2 = TimeDistributed(Convolution2D(256, (3, 3), strides = (2,2), padding='same', activation='relu'), name= 'feat2_td_conv2')(out_roi_pool_vgg2)   

    out_roi_pool_vgg4 = TimeDistributed(Convolution2D(256, (1, 1), strides = (1,1), padding='same', activation='relu'), name= 'feat4_td_conv1')(out_roi_pool_vgg4)

    # bp()

    out_roi_pool =merge([out_roi_pool_vgg1, out_roi_pool_vgg2, out_roi_pool_vgg4], mode='concat')
    #out_roi_pool = Lambda(function=lambda x: K.concatenate(x, axis = -1))([out_roi_pool_vgg1, out_roi_pool_vgg2, out_roi_pool_vgg4])

    out_roi_pool = TimeDistributed(Convolution2D(192, (1, 1), activation='relu', padding='same'), name='merge_conv_192')(out_roi_pool)
    out_roi_pool = TimeDistributed(Flatten())(out_roi_pool)

    [face_pred, pose_pred, gender_pred, viz_pred, landmark_pred, regr_pred] = classifier_layers(out_roi_pool, trainable)

    return([face_pred,pose_pred, gender_pred, viz_pred, landmark_pred, regr_pred])