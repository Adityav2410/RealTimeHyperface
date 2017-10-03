import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam,Nadam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
from keras_frcnn import resnet_hyper as nn
import keras_frcnn.roi_helpers as roi_helpers
from keras.applications.vgg16 import VGG16
import os
from pdb import set_trace as bp
from datetime import datetime

sys.setrecursionlimit(40000)

def parseOptions(C):

	parser = OptionParser()

	parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
	parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
					default="pascal_voc"),
	parser.add_option("-n", "--num_rois", dest="num_rois",
					help="Number of ROIs per iteration. Higher means more memory use.", default=32)
	parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=true).", action="store_true", default=False)
	parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
	parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
					  action="store_true", default=False)
	parser.add_option("--num_epochs", dest="num_epochs", help="Number of epochs.", default=2000)
	parser.add_option("--config_filename", dest="config_filename", help=
					"Location to store all the metadata related to the training (to be used when testing).",
					default="config.pickle")
	parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_hyper.hdf5')
	parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")

	(options, args) = parser.parse_args()

	if not options.train_path:   # if filename is not given
		parser.error('Error: path to training data must be specified. Pass --path to command line')

	if options.parser == 'pascal_voc':
		from keras_frcnn.pascal_voc_parser import get_data
	elif options.parser == 'simple':
		from keras_frcnn.simple_parser import get_data
	else:
		raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")



	C.num_rois = int(options.num_rois)
	C.use_horizontal_flips = bool(options.horizontal_flips)
	C.use_vertical_flips = bool(options.vertical_flips)
	C.rot_90 = bool(options.rot_90)

	C.model_path = options.output_weight_path

	if options.input_weight_path:
		C.base_net_weights = options.input_weight_path
	return options

	#all_imgs, classes_count, class_mapping = get_data(options.train_path)


def loadData(C):
	with open(C.data_annotation_path, 'rb') as f:
        aflw_annotation = pickle.load(f)
       return aflw_annotation

	all_imgs = aflw_annotation['all_imgs']
	class_mapping = aflw_annotation['class_mapping']
	classes_count = aflw_annotation['classes_count']

	if 'bg' not in classes_count:
		classes_count['bg'] = 0
		class_mapping['bg'] = len(class_mapping)
	inv_map = {v: k for k, v in class_mapping.iteritems()}

	C.class_mapping = class_mapping
	C.classes_count = classes_count
	C.inv_map = inv_map

	print( "Number of images: ",len(all_imgs))
	print('Training images per class:')
	pprint.pprint(classes_count)
	print('Num classes (including bg) = {}'.format(len(classes_count)))

	config_output_filename = options.config_filename
	with open(config_output_filename, 'w') as config_f:
		pickle.dump(C,config_f)
		print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

	#random.shuffle(all_imgs)
	return([all_imgs,class_mapping, classes_count, inv_map])


def getLogFileHandle()""
	log_filename = base_log_dir + '/trainHyper_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.log'
	fHandle = open(log_filename,'w')
	return fHandle


def getDatagenerator(all_imgs,C):
	num_imgs = len(all_imgs)

	train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
	val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

	print('Num train samples {}'.format(len(train_imgs)))
	print('Num val samples {}'.format(len(val_imgs)))

	data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_dim_ordering(), mode='train')
	data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, K.image_dim_ordering(), mode='val')

	return data_gen_train, data_gen_val


####################################### Build the model #################################################3


def getModel(C):
	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
	else:
		input_shape_img = (None, None, 3)


	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(C.num_rois, 4))

	base_model = VGG16(input_tensor = img_input, weights='imagenet', include_top=False) 

	# define the RPN, built on the base layers
	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn = nn.rpn(base_model.layers[-2].output, num_anchors)


	b1_feat = base_model.get_layer('block1_pool').output
	b2_feat = base_model.get_layer('block2_pool').output
	b4_feat = base_model.get_layer('block4_pool').output

	# Fuse features from 1st, 3rd and 4th conv blocks
	# classifier output: ([face_pred,pose_pred, gender_pred, viz_pred, landmark_pred, regr_pred])

	classifier = nn.classifier(b1_feat, b2_feat, b4_feat, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)
	model_classifier = Model([img_input,roi_input],classifier)


	model_rpn = Model(base_model.input, rpn[:2])

	feat_1_inp = Input(shape=None,None,64)
	feat_2_inp = Input(shape=None,None,128)
	feat_4_inp = Input(shape=None,None,512)
	classifier_only  = nn.classifier(feat_1_inp, feat_2_inp, feat_4_inp, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

	model_classifier_only = Model([feat1_inp, feat_2_inp, feat_4_inp, roi_input ], classifier_only )
	model_classifier_only = Model( [ b1_feat, b2_feat, b4_feat ] , classifier)







	model_classifier = Model([img_input, roi_input], classifier )

	classifier_only = 
	model_classifier_only = Model([img_input,b1_feat, b2_feat, b4_feat, roi_input ], classifier )




# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
# 	# print('loading weights from {}'.format(C.base_net_weights))
# 	# model_shared.load_weights()
	# print('loading RPN weights')
	# model_rpn.load_weights(C.base_net_weights, by_name=True)   #TODO: load RPN weights
# 	# model_classifier.load_weights(C.base_net_weights, by_name=True)
	print('loading weights from {}'.format(C.model_path))
	model_all.load_weights(C.model_path, by_name = True)
except:
	print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
		'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
		'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
	))

optimizer_rpn = Nadam(lr=1e-7)
optimizer_classifier = Nadam(lr=1e-7)
model_rpn.compile(optimizer=optimizer_rpn, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])

classifier_loss = [losses.class_loss_face(),losses.class_loss_pose(),losses.class_loss_gender(), losses.class_loss_viz(),losses.class_loss_landmark()]
classifier_loss_weight = [C.lambda_face,C.lambda_pose,C.lambda_gender,C.lambda_viz,C.lambda_landmark]
model_classifier.compile(optimizer=optimizer_classifier, loss=classifier_loss , loss_weights= classifier_loss_weight )

model_all.compile(optimizer='sgd', loss='mae')


##################################################################Training configuration ##########################################################
epoch_length = 100
num_epochs = int(options.num_epochs)
num_epochs = 210 * 10
iter_num = 0
epoch_num = 0

losses = np.zeros((epoch_length, 7))
overall_loss = np.zeros((epoch_length, 2))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.iteritems()}

#####################################################################################################################################################

print('Starting training')


def sliceMatrix(Y2,sel_samples, startIndex, endIndex):
	col1 = Y2[:,sel_samples,0].reshape((1,len(sel_samples),1))
	col_rest = Y2[:,sel_samples, startIndex:endIndex]
	Y = np.concatenate((col1, col_rest),axis=-1)
	return Y


while True:
	try:

		if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
			mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
			rpn_accuracy_rpn_monitor = []
			print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
			if mean_overlapping_bboxes == 0:
				print('RPN is not producing bounding boxes that overlap the ground truth boxes. Results will not be satisfactory. Keep training.')

		X, Y, img_data = data_gen_train.next()

		loss_rpn = model_rpn.train_on_batch(X, Y)



		P_rpn = model_rpn.predict_on_batch(X)

		R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)

		# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
		X2, Y2 = roi_helpers.calc_iou(R, img_data, C, class_mapping)

		if X2 is None:
			rpn_accuracy_rpn_monitor.append(0)
			rpn_accuracy_for_epoch.append(0)
			continue

		neg_samples = np.where(Y2[0, :, 0] == 0)
		pos_samples = np.where(Y2[0, :, 0] == 1)

		if len(neg_samples) > 0:
			neg_samples = neg_samples[0]
		else:
			neg_samples = []

		if len(pos_samples) > 0:
			pos_samples = pos_samples[0]
		else:
			pos_samples = []

		rpn_accuracy_rpn_monitor.append(len(pos_samples))
		rpn_accuracy_for_epoch.append((len(pos_samples)))


		losses[iter_num, 0] = loss_rpn[1]
		losses[iter_num, 1] = loss_rpn[2]


		iter_num += 1

		if iter_num == epoch_length:
			loss_rpn_cls 		= 	np.mean(losses[:, 0])
			loss_rpn_regr 		= 	np.mean(losses[:, 1])

			mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
			rpn_accuracy_for_epoch = []

			print("Epoch: {} \t RPN loss:{:.4f}\t".format(epoch_num,loss_rpn[0]) )
			print('Mean bounding box:{:.4f} \t loss_rpn_cls:{:.4f} \t loss_rpn_regr:{:.4f} \t elapsed_time:{:.1f}\t'.format(mean_overlapping_bboxes, loss_rpn_cls, loss_rpn_regr, time.time() - start_time))

			iter_num = 0
			start_time = time.time()
			epoch_num += 1
			if epoch_num == 1 or loss_rpn[0] < best_loss:
				if C.verbose:
					print('Total loss decreased from {} to {}, saving weights'.format(best_loss,loss_rpn[0]))
				best_loss = loss_rpn[0]
				model_all.save_weights(C.model_path)
				model_rpn.save_weights(C.base_net_weights)

		if epoch_num == num_epochs:
			print('Training complete, exiting.')
			fHandle.close()

			sys.exit()
	except Exception as e:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		continue



