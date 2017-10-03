import numpy as np
import math
import data_generators
import copy
from pdb import set_trace as bp

def calc_iou(R, img_data, C,class_mapping):
	# R = (boxes, probs)

	bboxes = img_data['bboxes'] # all the ground truthbboxes of one image
	(width, height) = (img_data['width'], img_data['height'])
	# get image dimensions for resizing
	(resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)

	gta = np.zeros((len(bboxes), 4))

	# Transform all the landmars into the resized image frame
	resize_ratio = (resized_width / float(width))/float(C.rpn_stride)
	sx, sy, sw, sh = C.classifier_regr_std

	for bbox_num, bbox in enumerate(bboxes):
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/C.rpn_stride))
		gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/C.rpn_stride))
		gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/C.rpn_stride))
		gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/C.rpn_stride))

	x_roi = []
	y_class_num = []
	y_class_regr_coords = []
	y_class_regr_label = []
	y_gender = []
	y_pose = []
	y_viz = []
	y_landmark = []
	

	# For each predicted box find the gt box that best overlaps above a threshold iou
	for ix in range(R.shape[0]):  # R.shape[0] = numboxes?
		# bp()
		(x1, y1, x2, y2) = R[ix, :]
		x1 = int(round(x1))
		y1 = int(round(y1))
		x2 = int(round(x2))
		y2 = int(round(y2))

		best_iou = 0.0
		best_bbox = -1
		# For the predicted(rpn) box <- iterate over all the ground truth box to find the best box
		for bbox_num in range(len(bboxes)):
			curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])
			if curr_iou > best_iou:
				best_iou = curr_iou
				best_bbox = bbox_num

		if best_iou < C.classifier_min_overlap:
				continue
		else:
			w = x2 - x1
			h = y2 - y1
			x_roi.append([x1, y1, w, h])
			pose_label = [0,0,0]
			gender_label = [0,0]
			viz_label = np.zeros(21)
			landmark_label = np.zeros(42)
			if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
				# hard negative example
				cls_name = 'bg'
				class_label = [0,1]
			elif C.classifier_max_overlap <= best_iou:
				cls_name = 'face'
				class_label = [1,0]
				pose_label = [ bbox['roll'],bbox['pitch'],bbox['yaw'] ]
				gender_label[ int(bbox['sex']=='f') ] = 1
				

				cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
				cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

				cx = x1 + w / 2.0
				cy = y1 + h / 2.0

				tx = (cxg - cx) / float(w)
				ty = (cyg - cy) / float(h)
				tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
				th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))

				#************ CHECK ************#
				viz_label  = bbox['feature_visible'] # Assuming a list of 21 ints

				# bp()
				trans_x = [sx*(xi - cx)*1./w for xi in (bbox['feature_x'] * resize_ratio) ]	# Transform the landmark coordinate from the original image to the feature map
				trans_y = [sy*(yi - cy)*1./h for yi in (bbox['feature_y'] * resize_ratio) ] 
				# bp()
			else:
				print('roi = {}'.format(best_iou))
				raise RuntimeError

		coords = [0] * 4
		landmark_label = [0] * 42

		y_class_num.append(copy.deepcopy(class_label))
		y_gender.append(copy.deepcopy(gender_label))
		y_pose.append(copy.deepcopy(pose_label))
		y_viz.append(copy.deepcopy(viz_label))
		# labels = [0] * 4 * (len(class_mapping) - 1)

		if cls_name != 'bg':
			# sx, sy, sw, sh = C.classifier_regr_std
			coords = [sx*tx, sy*ty, sw*tw, sh*th]
			y_class_regr_coords.append(copy.deepcopy(coords))

			landmark_label = trans_x + trans_y     # Assuming both are lists of 21 ints each
			y_landmark.append(copy.deepcopy(landmark_label))

		else:
			y_class_regr_coords.append(copy.deepcopy(coords))
			y_landmark.append(copy.deepcopy(landmark_label))


	if len(x_roi) == 0:
		return None, None, None

	X = np.array(x_roi)
	# Y1 = np.array(y_class_num)
	# Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)
	Y = np.concatenate([np.array(y_class_num), np.array(y_pose), np.array(y_gender), np.array(y_viz), np.array(y_landmark), np.array(y_class_regr_coords)],axis=-1)

	return[np.expand_dims(X, axis=0), np.expand_dims(Y, axis=0)]


def apply_regr(x, y, w, h, tx, ty, tw, th):
	try:
		cx = x + w/2.
		cy = y + h/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy
		w1 = math.exp(tw) * w
		h1 = math.exp(th) * h
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.
		x1 = int(round(x1))
		y1 = int(round(y1))
		w1 = int(round(w1))
		h1 = int(round(h1))

		return x1, y1, w1, h1

	except ValueError:
		return x, y, w, h
	except OverflowError:
		return x, y, w, h
	except Exception as e:
		print(e)
		return x, y, w, h

def apply_regr_np(X, T):
	try:
		x = X[0, :, :]
		y = X[1, :, :]
		w = X[2, :, :]
		h = X[3, :, :]

		tx = T[0, :, :]
		ty = T[1, :, :]
		tw = T[2, :, :]
		th = T[3, :, :]

		cx = x + w/2.
		cy = y + h/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy

		w1 = np.exp(tw) * w
		h1 = np.exp(th) * h
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.

		x1 = np.round(x1)
		y1 = np.round(y1)
		w1 = np.round(w1)
		h1 = np.round(h1)
		return np.stack([x1, y1, w1, h1])
	except Exception as e:
		print(e)
		return X

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# sort the bounding boxes 
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the intersection

		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])

		# find the union
		xx1_un = np.minimum(x1[i], x1[idxs[:last]])
		yy1_un = np.minimum(y1[i], y1[idxs[:last]])
		xx2_un = np.maximum(x2[i], x2[idxs[:last]])
		yy2_un = np.maximum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		ww_int = xx2_int - xx1_int
		hh_int = yy2_int - yy1_int

		ww_un = xx2_un - xx1_un
		hh_un = yy2_un - yy1_un

		ww_un = np.maximum(0, ww_un)
		hh_un = np.maximum(0, hh_un)

		# compute the ratio of overlap
		overlap = (ww_int*hh_int)/(ww_un*hh_un + 1e-9)

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlap_thresh)[0])))

		if len(pick) >= max_boxes:
			break

	# return only the bounding boxes that were picked using the integer data type
	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	# bp()
	return boxes, probs


"""
Takes all predicted bboxes and returns (boxes, probabilities); box coordinates in the RPN frame of reference
"""

def rpn_to_roi(rpn_layer, regr_layer, C, dim_ordering, use_regr=True, max_boxes=300,overlap_thresh=0.9):

	regr_layer = regr_layer / C.std_scaling

	anchor_sizes = C.anchor_box_scales
	anchor_ratios = C.anchor_box_ratios

	assert rpn_layer.shape[0] == 1

	if dim_ordering == 'th':
		(rows,cols) = rpn_layer.shape[2:]

	elif dim_ordering == 'tf':
		(rows, cols) = rpn_layer.shape[1:3]

	curr_layer = 0
	if dim_ordering == 'tf':
		A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
	elif dim_ordering == 'th':
		A = np.zeros((4, rpn_layer.shape[2], rpn_layer.shape[3], rpn_layer.shape[1]))

	for anchor_size in anchor_sizes:
		for anchor_ratio in anchor_ratios:

			anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride
			anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride
			if dim_ordering == 'th':
				regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :]
			else:
				regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
				regr = np.transpose(regr, (2, 0, 1))

			X, Y = np.meshgrid(np.arange(cols),np. arange(rows))

			A[0, :, :, curr_layer] = X - anchor_x/2
			A[1, :, :, curr_layer] = Y - anchor_y/2
			A[2, :, :, curr_layer] = anchor_x
			A[3, :, :, curr_layer] = anchor_y

			if use_regr:
				A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

			A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
			A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
			A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
			A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

			A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
			A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
			A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
			A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

			curr_layer += 1

	all_boxes = np.reshape(A.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0))
	all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

	x1 = all_boxes[:, 0]
	y1 = all_boxes[:, 1]
	x2 = all_boxes[:, 2]
	y2 = all_boxes[:, 3]

	# suppress invalid box coordinates
	idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

	all_boxes = np.delete(all_boxes, idxs, 0)
	all_probs = np.delete(all_probs, idxs, 0)
	# bp()

	# some kind of non maximum suppression. Does not use the ground truth. Keep <=300 boxes
	result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

	return result






def non_max_suppression_fast_classifier(boxes,bboxes_rpn, probs,poses, genders,vizs, landmrks, overlap_thresh=0.5,max_boxes=300):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# sort the bounding boxes 
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the intersection

		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])

		# find the union
		xx1_un = np.minimum(x1[i], x1[idxs[:last]])
		yy1_un = np.minimum(y1[i], y1[idxs[:last]])
		xx2_un = np.maximum(x2[i], x2[idxs[:last]])
		yy2_un = np.maximum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		ww_int = xx2_int - xx1_int
		hh_int = yy2_int - yy1_int

		ww_un = xx2_un - xx1_un
		hh_un = yy2_un - yy1_un

		ww_un = np.maximum(0, ww_un)
		hh_un = np.maximum(0, hh_un)

		# compute the ratio of overlap
		overlap = (ww_int*hh_int)/(ww_un*hh_un + 1e-9)

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlap_thresh)[0])))

		if len(pick) >= max_boxes:
			break

	# return only the bounding boxes that were picked using the integer data type
	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	poses = poses[pick]
	genders = genders[pick]
	vizs = vizs[pick] 
	landmrks = landmrks[pick]
	bboxes_rpn = bboxes_rpn[pick]
	return [boxes,bboxes_rpn, probs,poses, genders,vizs, landmrks]
