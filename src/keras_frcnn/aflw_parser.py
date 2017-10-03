import sqlite3
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import os.path
import numpy as np

def get_data(input_path):

	all_imgs = []

	classes_count = {}

	class_mapping = {}
	visualise = False

	conn = sqlite3.connect('../../aflw.sqlite')
	c = conn.cursor()
	# sql query
	select_string = "faceimages.filepath, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h"
	from_string = "faceimages, faces, facepose, facerect"
	where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id"

	# Debug

	cursor = conn.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
	print(cursor.fetchall())


	query = "SELECT faceimages.width FROM faceimages, faces, facepose, facerect"
	cursor = conn.execute(query)
	i = 0
	print [description[0] for description in cursor.description]
	for row in cursor.execute(query):
		#print row
		i+=1
		if i > 2:
			break

	# Distict file ids
	query_file_id = "SELECT DISTINCT file_id FROM faceimages"
	print('Parsing annotation files')
	file_ids = []
	for row in c.execute(query_file_id):
		 file_ids.append(row)
	print file_ids[0:10]

	# Annotation data
	# for file_id in file_ids:
	# 	select_string = "faceimages.filepath, faceimages.file_id, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h"
	# 	from_string = "faceimages, faces, facepose, facerect"
	# 	#where_string = "faceimages.file_id = " + str(file_id) #+ " and faceimages.file_id = faces.file_id"

	# 	query_string = "SELECT " + select_string + " FROM " + from_string# + " WHERE " + where_string
	# 	annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,'height': element_height, 'bboxes': []}
	# 	annotation_data['image_set']
	# 	for row in c.execute(query_string):
	# 		print row[1]
	# 		annotation_data['bboxes'].append(
	# 					{'class' = 1, 'headpose': headpose, 'landmarks': landmarks, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'visibility': visibility, 
	# 					'gender'= gender})
	# 		break
	# 	all_imgs.append[annotation_data]
	# 	break
					


get_data('../../aflw.sqlite')


