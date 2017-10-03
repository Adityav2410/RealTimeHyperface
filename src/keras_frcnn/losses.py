from keras import backend as K
from keras.objectives import categorical_crossentropy,mean_squared_error
from pdb import set_trace as bp

if K.image_dim_ordering() == 'tf':
	import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

##################################################################################### RPN LOSS ####################################################################################
def rpn_loss_regr(num_anchors):
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		if K.image_dim_ordering() == 'th':
			x = y_true[:, 4 * num_anchors:, :, :] - y_pred
			x_abs = K.abs(x)
			x_bool = K.less_equal(x_abs, 1.0)
			return lambda_rpn_regr * K.sum(
				y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
		else:
			x = y_true[:, :, :, 4 * num_anchors:] - y_pred
			x_abs = K.abs(x)
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

			return lambda_rpn_regr * K.sum(
				y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

	return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		import time
		time.sleep(5)
		if K.image_dim_ordering() == 'tf':
			return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
		else:
			return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, num_anchors:, :, :])) / K.sum(epsilon + y_true[:, :num_anchors, :, :])

	return rpn_loss_cls_fixed_num




##################################################################################### CLASSIFIER LOSS ####################################################################################
def class_loss_regr():
	def class_loss_regr_fixed_num(y_true, y_pred):

		face_true = y_true[0,:,0]
		x = y_true[0,:,1:] - y_pred[0,:,:]
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return K.sum( K.expand_dims(face_true, axis = -1) * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + face_true)
	return class_loss_regr_fixed_num


def class_loss_face():
	def class_loss_face_fixed_sum(y_true, y_pred):
		face_true = y_true[0,:,0]
		num_face = K.sum(face_true)
		ll =  K.sum( categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]) )/32
		return ll
	return class_loss_face_fixed_sum

# Consider all the positive examples for the losses defined below
# Note: mean_squared_error by default takes mean along axis = -1
def class_loss_pose():
	def class_loss_pose_fixed_sum(y_true, y_pred):
		# bp()
		face_true = y_true[0,:,0]
		num_face = K.sum(face_true)
		return K.sum( face_true * mean_squared_error(y_true[0, :,1:], y_pred[0, :, :]))/32
	return class_loss_pose_fixed_sum


def class_loss_gender():
	def class_loss_gender_fixed_sum(y_true, y_pred):
		face_true = y_true[0,:,0]
		num_face = K.sum(face_true)
		return K.sum( face_true * categorical_crossentropy(y_true[0, :, 1:], y_pred[0, :, :]))/32
	return class_loss_gender_fixed_sum


def class_loss_viz():
	def class_loss_viz_fixed_sum(y_true, y_pred):
		face_true = y_true[0,:,0]
		num_face = K.sum(face_true)
		return K.sum( face_true * mean_squared_error(y_true[0,:,1:] , y_pred[0,:,:]))/(32*42)
	return class_loss_viz_fixed_sum


def class_loss_landmark():
	def class_loss_landmark_fixed_sum(coord_true, coord_pred):
		viz_true = coord_true[0,:,1:22]
		# bp()

		x_true_coord = coord_true[0,:,22:43]
		y_true_coord = coord_true[0,:,43:64]
		
		x_pred_coord = coord_pred[0,:,0:21]
		y_pred_coord = coord_pred[0,:,21:42]

		num_viz_feature = K.sum(viz_true)
		return K.sum(viz_true * (K.square(x_true_coord - x_pred_coord) + K.square(y_true_coord - y_pred_coord)), axis=-1)/(32*42) #(num_viz_feature+0.01)
		#return K.sum(viz_true[0,:,:] * (K.square(y_true_x[0,:,:] - y_pred_x[0,:,:]) + K.square(y_true_y[0,:,:] - y_pred_y[0,:,:])), axis=-1)/(num_viz_feature+0.01)
	return class_loss_landmark_fixed_sum