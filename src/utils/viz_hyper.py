import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam
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

import cv2
import pandas as pd
from pdb import set_trace as bp



sys.setrecursionlimit(40000)

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


with open('aflw_annotation.pkl', 'rb') as f:
        aflw_annotation = pickle.load(f)

log_filename = './log_dir/trainHyper_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.log'
fHandle = open(log_filename,'w')

all_imgs = aflw_annotation['all_imgs']
C = config.Config()

C.num_rois = int(options.num_rois)
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.model_path = options.output_weight_path
# C.model_path = 

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

####################################### Build the model #################################################3

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (None, None, 3)



img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
b1_inp = Input(shape = (None, None,64),name="feature1_input")
b2_inp = Input(shape = (None, None,128),name="feature2_input")
b4_inp = Input(shape = (None, None,512),name="feature4_input")



# define the base network (resnet here, can be VGG, Inception, etc)
#shared_layers = nn.nn_base(img_input, trainable=True)
base_model = VGG16(input_tensor = img_input, weights='imagenet', include_top=False) 
b1_feat = base_model.get_layer('block1_conv2').output
b2_feat = base_model.get_layer('block2_pool').output
b4_feat = base_model.get_layer('block4_pool').output


# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn( b1_feat, b2_feat, b4_feat, num_anchors)
model_rpn = Model(base_model.input, rpn )

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
# model_all = Model([img_input, roi_input], rpn[:2] + classifier)
classifier_only = nn.classifier( b1_inp, b2_inp, b4_inp, roi_input, C.num_rois, nb_classes=2, trainable=True )
model_classifier_only = Model([b1_inp, b2_inp, b4_inp, roi_input], classifier_only)
# # classifier = [face_out,pose_out, gender_out, viz_out, landmark_out, regr_out]



try:
	print('loading classifier weights from {}'.format(C.model_path))
	model_classifier_only.load_weights(C.model_path, by_name = True)
	print('loading rpn weights from {}'.format(C.model_path))
	model_rpn.load_weights(C.model_path, by_name=True)   #TODO: load RPN weights
	# model_rpn.load_weights('./model_rpn.hdf5', by_name=True)   #TODO: load RPN weights


except:
	print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
		'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
		'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
	))


model_rpn.compile(optimizer='sgd',loss='mse')
model_classifier_only.compile(optimizer='sgd', loss='mse' )

#################################################### VISUALIZE IMAGES ##############################################33
# img_path = '../viz_data/viz_images'
# img_save_path = '../viz_data/viz_annotated_images'

# img_path = '../viz_data/viz_images'
img_path = '../viz_data/viz_images_2'
img_save_path = '../viz_data/viz_annotated_images/'
# img_save_path = '../viz_data/viz_annotated_landmark/'
# img_save_path = '../viz_data/viz_annotated_regr_improve/'
files =  os.listdir(img_path)
files.sort()

bbox_threshold = 0.8


def format_img(img, C):
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	
	if width <= height:
		f = img_min_side/width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side/height
		new_width = int(f * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	# img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img


# bp()

overallStartTime = time.time()
file_save_count = 0
for img_name in files:
	# img_name = 'image06785.jpg'
	try:
		start_time = time.time()

		filepath = os.path.join(img_path,img_name)
		img = cv2.imread(filepath)
		print("Processing file: ", img_name)
		X = format_img(img, C)

		img_scaled = np.transpose(X.copy()[0, (2, 1, 0), :, :], (1, 2, 0)).copy()
		img_scaled[:, :, 0] += 123.68
		img_scaled[:, :, 1] += 116.779
		img_scaled[:, :, 2] += 103.939
		img_scaled = img_scaled.astype(np.uint8)

		if K.image_dim_ordering() == 'tf':
			X = np.transpose(X, (0, 2, 3, 1))

		# get the feature maps and output from the RPN
		rpnStartTime = time.time()
		[Y1, Y2, feat1, feat2, feat4 ] = model_rpn.predict(X)
		print("RPN prediction time:", time.time()-rpnStartTime)
		# bp()

		R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
		# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]

		bboxes = []
		bboxes_rpn = []
		probs = []
		genders = []
		poses = []
		landmarks = []
		vizs = []

		sx = C.classifier_regr_std[0]
		sy = C.classifier_regr_std[1]

		classifierStartTime = time.time()
		# C.num_rois = 32
		for jk in range(R.shape[0]//C.num_rois + 1):
			ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
			if ROIs.shape[1] == 0:
				break

			if jk == R.shape[0]//C.num_rois:
				#pad R
				curr_shape = ROIs.shape
				target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
				ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
				ROIs_padded[:, :curr_shape[1], :] = ROIs
				ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
				ROIs = ROIs_padded

			# bp()
			[face_pred,pose_pred,gender_pred,viz_pred,land_pred,regr_pred] = model_classifier_only.predict([feat1, feat2, feat4 , ROIs])

			for ii in range(face_pred.shape[1]):

				if np.max(face_pred[0, ii, :]) < bbox_threshold or np.argmax(face_pred[0, ii, :]) == (face_pred.shape[2] - 1):
					continue
				
				############### bBox regression ######################

				(x, y, w, h) = ROIs[0, ii, :]

				bboxes_rpn.append([16*x, 16*y, 16*(x+w), 16*(y+h)])
	

				probs.append(face_pred[0, ii, 0])
				poses_int = pose_pred[0,ii,:]*180/3.14 
				poses.append( poses_int.astype(int) )
				genders.append( 'f' if np.argmax(gender_pred[0,ii,:]) else 'm')
				vizs.append(viz_pred[0,ii,:])

				# bp()
				xc = x + w/2.0
				yc = y + w/2.0
				land_pred[0,ii,:21] /= sx
				land_pred[0,ii,21:] /= sy
				land_pred[0,ii,:21] = 16*( land_pred[0,ii,:21]*w + xc )
				land_pred[0,ii,21:] = 16*( land_pred[0,ii,21:]*h + yc )
				landmarks.append(land_pred[0,ii,:].astype(int) )

				# bp
				try:
					(tx, ty, tw, th) = regr_pred[0, ii, : ]
					tx /= C.classifier_regr_std[0]
					ty /= C.classifier_regr_std[1]
					tw /= C.classifier_regr_std[2]
					th /= C.classifier_regr_std[3]
					x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
				except:
					pass
				bboxes.append([16*x, 16*y, 16*(x+w), 16*(y+h)])




		# bp()
		print("Classifier prediction time:", time.time()-classifierStartTime)
		all_dets = []
		# bp()
		[new_bboxes,new_bboxes_rpn, new_probs,new_poses, new_genders,new_vizs, new_landmarks] = roi_helpers.non_max_suppression_fast_classifier(
																				np.array(bboxes),np.array(bboxes_rpn),np.array(probs),np.array(poses), np.array(genders),np.array(vizs), 
																				np.array(landmarks), overlap_thresh=0.2)
		# bp()
		for jk in range(new_bboxes.shape[0]):
			# bp()
			(x1, y1, x2, y2) = new_bboxes[jk,:]
			(x1_rpn, y1_rpn, x2_rpn, y2_rpn) = new_bboxes_rpn[jk,:]


			cv2.rectangle(img_scaled,(x1, y1), (x2, y2), (0,0,255) if new_genders[jk] == 'f' else (255,0,0), 2)
			cv2.rectangle(img_scaled,(x1_rpn, y1_rpn), (x2_rpn, y2_rpn), (0,225,255) )

			#textLabel = '{}'.format(new_genders[jk])
			textLabel = '{}: {} : {}'.format(new_poses[jk][0], new_poses[jk][1], new_poses[jk][2])
			textLabel2 = '{}:     {:.2f}'.format( new_genders[jk], new_probs[jk] )
			#all_dets.append((key,100*new_probs[jk]))

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_PLAIN,1,1)
			(retval2,baseLine2) = cv2.getTextSize(textLabel2,cv2.FONT_HERSHEY_PLAIN,1,1)
			textOrg = (x1, y1)
			textOrg2 = ( x1, y2 )

			cv2.rectangle(img_scaled, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img_scaled, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)

			cv2.rectangle(img_scaled, (textOrg2[0] - 5, textOrg2[1]+baseLine2 - 5), (textOrg2[0]+retval2[0] + 5, textOrg2[1]-retval2[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img_scaled, (textOrg2[0] - 5, textOrg2[1]+baseLine2 - 5), (textOrg[0]+retval2[0] + 5, textOrg2[1]-retval2[1] - 5), (255, 255, 255), -1)

			cv2.putText(img_scaled, textLabel, textOrg, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
			cv2.putText(img_scaled, textLabel2, textOrg2, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
			
			for i in range(21):
				if new_vizs[jk,i] > 0.7 : 
					cv2.circle(img_scaled, (new_landmarks[jk,i], new_landmarks[jk,i+21]), 2, (0,255,0))

			print "Predicted orientation", textLabel



		# bp()
		# for i_face in data_annotation[img_name]['face'].values():
		# 	orient = i_face['orientation']#['orientation']
		# 	# orient = data_annotation[img_name]['face'].values()[0]['orientation']#['orientation']
		# 	orient_2 = (180/3.14) * np.array(orient) 
		# 	orient_2.astype('int')
		# 	print "True orientation", orient_2


		print('Elapsed time = {}'.format(time.time() - start_time))
		# cv2.imshow('img', img_scaled)
		# cv2.waitKey(0)
		cv2.imwrite( os.path.join(img_save_path,'img'+str(file_save_count)+'.jpeg') ,img_scaled)
		file_save_count += 1
		# filename = 'image71306.jpg'
		
	except Exception as e:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		continue

print("Video processing time: ", time.time()-overallStartTime)



	#print(all_dets)
###############################################