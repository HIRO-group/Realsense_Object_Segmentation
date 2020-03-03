
###################################################################
##      Intel Realsense image to binary segmentation image       ##
###################################################################

# AUTHOR : Jacob Fiola - HIRO robotics @ University of Colorado Boulder
# LAST UPDATED - Oct 20 2019
# PY VERSION: 3.7.3
# DESCRIPTION: This is an algorithm with inputs = (rs.pipeline)
#	and output = 640x480 binary segmentation image
#	pixels are determind to be within an object by considering these factors:
# 	-FACTOR 1: Difference in depth for each pixel when compared to their base depths (the first frame's depths)

# ~~~~~~TODO~~~~~~~~
# -FACTOR 2: Implement image subtraction for each frame.
# -FACTOR 3: Implement "Monocular depth estimation" - (A Convolutional Nerural Network)
# THEN, figure out how much we need to care about each factor. This may be constant, OR, this may be some probability distribution.



#import pyrealsense, numpy, and openCV
import pyrealsense2 as rs
import scipy.spatial as spp
import numpy as np
import cv2
import argparse
import skimage.segmentation as seg
import skimage.color as color



# Configure depth and color streams. This currently sets autoexposure to ON.
pipeline = rs.pipeline()
config = rs.config()
calibrating = True

pixelHeight = 480
pixelWidth = 848


numFramesToSkip = 30
numCalibrationFrames = 50



kernel = np.ones((3,3),np.uint8)
kernel2 = np.ones((2,2),np.uint8)
kernel3 = np.ones((3,3),np.uint8)

config.enable_stream(rs.stream.depth, pixelWidth, pixelHeight, rs.format.z16, 30)
config.enable_stream(rs.stream.color, pixelWidth, pixelHeight, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
device = profile.get_device()

#set exposure and other camera options
depth_sensor = profile.get_device().first_depth_sensor()

color_sensor = device.query_sensors()[1]
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
color_sensor.set_option(rs.option.exposure, 180)
color_sensor.set_option(rs.option.sharpness, 120)
print(color_sensor.get_supported_options())

depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
#depth_sensor.set_option(rs.option.gain, 16)
depth_sensor.set_option(rs.option.laser_power,360)
#depth_sensor.set_option(rs.option.frame_rate,30)


# Getting the depth sensor's depth scale
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

window1Name = 'input image (purposefully blurred)'
window2Name = 'Depth difference'
window3Name = 'RGB difference'
window4Name = "RGB and Depth difference union"
window5Name = "Watershed"
#window6Name = "test"

color_difference_threshold_darker = 150

alpha_slider_max = 100
title_window = 'RGB Darker threshold'

def empty(z):
	pass
#def on_trackbar(val):
#	alpha = val / alpha_slider_max
#	beta = ( 1.0 - alpha )
	#color_difference_threshold_darker = alpha*200
	#print(color_difference_threshold_darker)
v_difference_threshold_darker_default = 35
v_difference_threshold_lighter_default = 30
color_difference_threshold_lighter_default = 30
color_difference_threshold_darker_default = 53

def recalibrate(newVal):
	framesCaptured = 0
	calibrating = True

	#cv2.setTrackbarPos('Recalibrate', 'controls', 0)



cv2.namedWindow('controls',cv2.WINDOW_NORMAL)
cv2.resizeWindow("controls", 1000,100);
#cv2.createTrackbar('v lighter difference threshold', 'controls' , 0, 255, empty)
#cv2.createTrackbar('v darker difference threshold', 'controls' , 0, 255, empty)
cv2.createTrackbar('RGB lighter difference threshold', 'controls' , 0, 255, empty)
cv2.createTrackbar('RGB darker difference threshold', 'controls' , 0, 255, empty)
cv2.createTrackbar('Recalibrate', 'controls' , 0, 1, empty)

#cv2.setTrackbarPos('v darker difference threshold', 'controls', v_difference_threshold_darker_default)
#cv2.setTrackbarPos('v lighter difference threshold', 'controls', v_difference_threshold_lighter_default)
cv2.setTrackbarPos('RGB lighter difference threshold', 'controls', color_difference_threshold_lighter_default)
cv2.setTrackbarPos('RGB darker difference threshold', 'controls', color_difference_threshold_darker_default)

# We will be removing the background of objects more than
# clipping_distance_in_meters meters away
#clipping_distance_in_meters = 2.1
#clipping_distance_in_meters_top = 0
#clipping_distance = clipping_distance_in_meters / depth_scale
#clipping_distance_top = clipping_distance_in_meters_top / depth_scale


# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


# Make windows
cv2.namedWindow(window1Name, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(window1Name,0,20)
cv2.moveWindow("controls",2000,20)

cv2.namedWindow(window2Name, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(window2Name,0,800)

cv2.namedWindow(window3Name, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(window3Name,1000,20)

cv2.namedWindow(window4Name, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(window4Name,1000,800)

cv2.namedWindow(window5Name, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(window5Name,1000,1300)

#cv2.namedWindow(window6Name, cv2.WINDOW_AUTOSIZE)
#cv2.moveWindow(window6Name,0,1400)

#cv2.namedWindow('thresh of unculled color image', cv2.WINDOW_AUTOSIZE)
#cv2.moveWindow('thresh of unculled color image',1500,20)
#cv2.namedWindow('thresh of culled color image', cv2.WINDOW_AUTOSIZE)
#cv2.moveWindow('thresh of culled color image',1500,620)
#cv2.namedWindow('higher constant being subtracted', cv2.WINDOW_AUTOSIZE)
#cv2.moveWindow('higher constant being subtracted',1500,1220)


calibration_image = cv2.imread('calibrating.png',1)
ignoring_image = cv2.imread('ignoring.png',1)


framesCaptured = 0
numRows = pixelHeight
numColumns = pixelWidth
numPixelsNotZero = [0 for k in range(numRows)] # define these outside the forever loop
for i in range(len(numPixelsNotZero)):
	numPixelsNotZero[i] = [0 for k in range(numColumns)]


firstFrameCaptured = False # Have we captured and stored the data for the first frame yet?
test = True #for debugging
try:
	while True:

		# Wait for a coherent pair of frames: depth and color
		frames = pipeline.wait_for_frames()

		# Align the depth frame to color frame
		aligned_frames = align.process(frames)

		# Get aligned frames. Aligned_depth_frame is a 640x480 depth image
		aligned_depth_frame = aligned_frames.get_depth_frame()
		color_frame = aligned_frames.get_color_frame()


        # Validate that both frames are valid
		if not aligned_depth_frame or not color_frame:
			continue

		#get depth/color data and put into a numpy arrays
		depth_image = np.asanyarray(aligned_depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())
		hsv_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)
		#color_image_blurred = cv2.blur(color_image,(3,3))
		b,g,r = cv2.split(color_image)


		#h,s,v = cv2.split(hsv_image)

		calButton = cv2.getTrackbarPos('Recalibrate','controls')
		if calButton == 1:
			framesCaptured = False
			calibrating = True
			cv2.setTrackbarPos('Recalibrate','controls',0)


		#if we haven't seen an image yet
		if(calibrating):
			#print("calibrating")
			if framesCaptured < numFramesToSkip:
				framesCaptured +=1
			elif framesCaptured ==numFramesToSkip:
				firstDepthFrame = depth_image.astype(np.uint32)
				numFramesNotZeroForPixel = np.where(depth_image > 0,1,0)
				bInitial,gInitial,rInitial = b,g,r
				#hInitial,sInitial,vInitial = h,s,v
				framesCaptured +=1
			elif framesCaptured < numFramesToSkip+numCalibrationFrames:
				firstDepthFrame = np.add(firstDepthFrame.astype(np.uint32),depth_image.astype(np.uint32))
				bInitial = np.add(bInitial.astype(np.uint32),b.astype(np.uint32))
				gInitial = np.add(gInitial.astype(np.uint32),g.astype(np.uint32))
				rInitial = np.add(rInitial.astype(np.uint32),r.astype(np.uint32))
				#hInitial = np.add(hInitial.astype(np.uint32),h.astype(np.uint32))
				#sInitial = np.add(sInitial.astype(np.uint32),s.astype(np.uint32))
				#vInitial = np.add(vInitial.astype(np.uint32),v.astype(np.uint32))
				numFramesNotZeroForPixel = np.where(depth_image > 0,numFramesNotZeroForPixel+1,numFramesNotZeroForPixel)
				framesCaptured +=1
			else:
				averageDepthFrame = np.divide(firstDepthFrame,np.where(numFramesNotZeroForPixel>0,numFramesNotZeroForPixel,1))
				#print("b Initial before division",bInitial[0][0])
				averageDepth = np.average(averageDepthFrame)
				averageDepthInMeters = depth_scale*averageDepth
				averageDepthInMM = averageDepthInMeters*1000
				bInitial = np.divide(bInitial,numCalibrationFrames).astype(np.uint8)
				gInitial = np.divide(gInitial,numCalibrationFrames).astype(np.uint8)
				rInitial = np.divide(rInitial,numCalibrationFrames).astype(np.uint8)
				#hInitial = np.divide(hInitial,numCalibrationFrames).astype(np.uint8)
				#sInitial = np.divide(sInitial,numCalibrationFrames).astype(np.uint8)
				#vInitial = np.divide(vInitial,numCalibrationFrames).astype(np.uint8)

				averageR = np.average(rInitial)
				averageG = np.average(gInitial)
				averageB = np.average(bInitial)
				euclidean_length = np.sqrt(averageR**2+averageG**2+averageB**2)
				color_difference_threshold_lighter_default = np.max([(1000/8333)*euclidean_length + (63324/83333),3])
				color_difference_threshold_darker_default = np.max([(2400/8333)*euclidean_length + (33649/83333),5])
				cv2.setTrackbarPos('RGB lighter difference threshold','controls',int(color_difference_threshold_lighter_default))
				cv2.setTrackbarPos('RGB darker difference threshold','controls',int(color_difference_threshold_darker_default))

				print("r: " + str(averageR)+ " g: " + str(averageG) + " b: " + str(averageB))
				print("euclidean length: " + str(euclidean_length))
				print("RGB lighter threshold: " + str(color_difference_threshold_lighter_default))
				print("RGB darker threshold: " + str(color_difference_threshold_darker_default))
				#print("b Initial after division",bInitial[0][0])
				#initialColors = np.sqrt(np.add(np.multiply(bInitial,bInitial),np.add(np.multiply(gInitial,gInitial),np.multiply(rInitial,rInitial)))).astype(np.uint8)
				framesCaptured +=1
				if averageDepthInMeters > 2:
					depth_difference_threshold = 9999
					ignoring_depth = True
				else:
					depth_difference_threshold = (averageDepthInMM**2)/9000
					print("depth difference threshhold " + str(depth_difference_threshold))
					ignoring_depth = False

				calibrating = False

				# Show images
			cv2.imshow(window1Name, calibration_image)

		else:
			depth_image = np.where(depth_image > 0, depth_image,averageDepthFrame)
			depth_diffs = cv2.subtract(averageDepthFrame,depth_image)
			#v_difference_threshold_darker=cv2.getTrackbarPos('v darker difference threshold', 'controls')
			#v_difference_threshold_lighter=cv2.getTrackbarPos('v lighter difference threshold', 'controls')
			color_difference_threshold_darker = cv2.getTrackbarPos('RGB darker difference threshold', 'controls')
			color_difference_threshold_lighter = cv2.getTrackbarPos('RGB lighter difference threshold', 'controls')
			#if(vDarker!= 0):
			#else:
			#	v_difference_threshold_darker = 35
			#if(vLighter != 0):
			#else:
			#	v_difference_threshold_darker = 30
			#if (hul != -1):
			#	v_difference_threshold_darker = hul
			#	print(hul)
			#else:
			#	v_difference_threshold_darker = 35



			#currColors = np.sqrt(np.add(np.multiply(b,b),np.add(np.multiply(g,g),np.multiply(r,r)))).astype(np.uint8)
			#color_diffs = np.sqrt(np.add(np.square(np.absolute(np.subtract(b,bInitial))),np.add(np.square(np.absolute(np.subtract(g,gInitial))),np.square(np.absolute(np.subtract(r,rInitial))))))
			#print(color_diffs[0][0])
			#print("Average Depth:" + str(averageDepthInMeters))
			#image_felz = seg.felzenszwalb(color_image)
			#image_felzenszwalb_colored = color.label2rgb(image_felz, color_image, kind='avg')
			#image_slic = seg.slic(color_image,n_segments=155)
			#slic = color.label2rgb(image_slic, color_image, kind='avg')
			gray = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)


			blue_diffs_lighter = cv2.subtract(b,bInitial)

			blue_diffs_darker = cv2.subtract(bInitial,b)
			green_diffs_lighter = cv2.subtract(g,gInitial)
			green_diffs_darker = cv2.subtract(gInitial,g)
			red_diffs_lighter = cv2.subtract(r,rInitial)
			red_diffs_darker = cv2.subtract(rInitial,r)

			ret, thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

			#ret, thresh2 = cv2.threshold(blue_diffs_darker,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			#ret, thresh3 = cv2.threshold(green_diffs_lighter,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			#ret, thresh4 = cv2.threshold(green_diffs_darker,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			#ret, thresh5 = cv2.threshold(red_diffs_lighter,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			#ret, thresh6 = cv2.threshold(red_diffs_darker,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

			#hdiffs_lighter = cv2.subtract(h,vInitial)
			#hdiffs_darker = cv2.subtract(hInitial,h)

			#sdiffs_lighter = cv2.subtract(s,vInitial)
			#sdiffs_darker = cv2.subtract(sInitial,s)

			#vdiffs_lighter = cv2.subtract(v,vInitial)
			#vdiffs_darker = cv2.subtract(vInitial,v)

			#blue_diffs = cv2.absdiff(b,bInitial)
			#green_diffs = cv2.absdiff(g,gInitial)
			#red_diffs = cv2.absdiff(r,rInitial)


			#if test: #this is just debugging info from the first frame
				#print(depth_image[0])
				#print(len(depth_image[0]))
				#print(np.max(depth_image))
				#print(green_diffs[0][0])
				#print(depth_diffs[0])
				#print(culled_color_image[0][0])
				#print(len(color_image)==len(culled_color_image))
			#	test = False

			#if there is a difference greater


			 #the minimum difference in depth to care about
			#print(averageDepthInMeters)
			#h_difference_threshold = 35
			#s_difference_threshold = 50
			#h_difference_threshold_darker = 20
			#h_difference_threshold_lighter = 30

			#s_difference_threshold_darker = 1000
			#s_difference_threshold_lighter = 10


			#print(v_difference_threshold_darker)
			#v_difference_threshold_lighter = 30

			#print(color_difference_threshold_darker)
			#color_difference_threshold_darker = 130
			#color_difference_threshold_test = 155
			#color_difference_threshold_test1 = 50
			#color_difference_threshold_test2 = 50

			white_color = 30000       #in COLORMAP_BONE, a really high number corresponds to white.
			black_color = 0           #in COLORMAP_BONE


			binary_depth_image = np.where(depth_diffs > depth_difference_threshold,white_color,black_color)
			#culled_color_image = np.linalg.norm(color)
			culled_color_image = np.where(blue_diffs_lighter > color_difference_threshold_lighter,white_color,black_color)
			culled_color_image = np.where(green_diffs_lighter > color_difference_threshold_lighter,white_color,culled_color_image)
			culled_color_image = np.where(red_diffs_lighter > color_difference_threshold_lighter,white_color,culled_color_image)

			culled_color_image = np.where(blue_diffs_darker > color_difference_threshold_darker,white_color,culled_color_image)
			culled_color_image = np.where(green_diffs_darker > color_difference_threshold_darker,white_color,culled_color_image)
			culled_color_image = np.where(red_diffs_darker > color_difference_threshold_darker,white_color,culled_color_image)


			#culled_hsv_image = np.where(hdiffs > h_difference_threshold,white_color,black_color)
			#culled_hsv_image = np.where(sdiffs > s_difference_threshold,white_color,black_color)
			#culled_hsv_image = np.where(vdiffs_lighter > v_difference_threshold_lighter,white_color,black_color)
			#culled_hsv_image = np.where(vdiffs_darker > v_difference_threshold_darker,white_color,culled_hsv_image)

			#culled_hsv_image = np.where(sdiffs_lighter > s_difference_threshold_lighter,white_color,black_color)
			#culled_hsv_image = np.where(sdiffs_darker > s_difference_threshold_darker,white_color,culled_hsv_image)
			thresh_image = np.where(thresh1>1,white_color,black_color)
			#thresh_image = np.where(thresh2>1,white_color,thresh_image)
			#thresh_image = np.where(thresh3>1,white_color,thresh_image)
			#thresh_image = np.where(thresh4>1,white_color,thresh_image)
			#thresh_image = np.where(thresh5>1,white_color,thresh_image)
			#thresh_image = np.where(thresh6>1,white_color,thresh_image)



			#print("binitial",bInitial[0][0])
			#print("bcurr",b[0][0])

			#culled_color_image = np.where(blue_diffs_darker > color_difference_threshold_darker,white_color,culled_color_image)
			#clled_color_image = np.where(green_diffs_darker > color_difference_threshold_darker,white_color,culled_color_image)
			#culled_color_image = np.where(red_diffs_darker > color_difference_threshold_darker,white_color,culled_color_image)
			#culled_color_image = cv2.blur(culled_color_image,(10,10))

			#combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_OPEN, kernel2)
			#combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_CLOSE, kernel2)




			#print(red_diffs_darker[0][0])

			#culled_color_image = np.where(blue_diffs > color_difference_threshold_test1,white_color,black_color)
			#culled_color_image = np.where(green_diffs > color_difference_threshold_test1,white_color,culled_color_image)
			#culled_color_image = np.where(red_diffs > color_difference_threshold_test1,white_color,culled_color_image)
			#culled_color_image = np.where(blue_diffs > color_difference_threshold_test,black_color,culled_color_image)
			#culled_color_image = np.where(green_diffs > color_difference_threshold_test,black_color,culled_color_image)
			#culled_color_image = np.where(red_diffs > color_difference_threshold_test,black_color,culled_color_image)

			#culled_color_image = np.where(color_diffs > color_difference_threshold_test2,white_color,black_color)

			#combined_image = np.where(culled_color_image == white_color,white_color,black_color)
			#combined_image = np.where(binary_depth_image == white_color,white_color,combined_image)

			#combined_diffs = np.add(blue_diffs,np.add(green_diffs,red_diffs))

			#culled_color_image = np.where(combined_diffs>200,white_color,black_color)

			#print("blue diff max", np.min(blue_diffs_darker))
			#print("green diff max", np.max(green_diffs_darker))
			#print("red diff max", np.max(red_diffs_darker))


			# Remove background - Set pixels further than clipping_distance to grey
			#grey_color = 153
			#depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
			#bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0) | (depth_image_3d < clipping_distance_top), grey_color, color_image)



			# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
			#hsv_colormap = cv2.applyColorMap(cv2.convertScaleAbs(culled_hsv_image, alpha=0.03), cv2.COLORMAP_BONE)
			depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(binary_depth_image, alpha=0.03), cv2.COLORMAP_BONE)
			depth_colormap = cv2.morphologyEx(depth_colormap, cv2.MORPH_OPEN, kernel)
			depth_colormap= cv2.morphologyEx(depth_colormap, cv2.MORPH_CLOSE, kernel)

			image_diff_colormap = cv2.applyColorMap(cv2.convertScaleAbs(culled_color_image, alpha=0.03), cv2.COLORMAP_BONE)
			#image_diff_colormap = cv2.morphologyEx(image_diff_colormap, cv2.MORPH_OPEN, kernel2)
			#image_diff_colormap = cv2.morphologyEx(image_diff_colormap, cv2.MORPH_OPEN, kernel2)

			test= np.where(image_diff_colormap > 0,white_color,black_color)
			test_colormap = cv2.applyColorMap(cv2.convertScaleAbs(test, alpha=0.03), cv2.COLORMAP_BONE)
			thresh_colormap = cv2.applyColorMap(cv2.convertScaleAbs(thresh_image, alpha=0.03), cv2.COLORMAP_BONE)

			combined = np.where(image_diff_colormap != 0, white_color,black_color)
			combined = np.where(depth_colormap != 0,white_color,combined)


			combined_colormap = cv2.applyColorMap(cv2.convertScaleAbs(combined, alpha=0.03), cv2.COLORMAP_BONE)
			combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_OPEN, kernel2)
			combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_CLOSE, kernel2)



			#image_diff_colormap2= cv2.applyColorMap(cv2.convertScaleAbs(culled_color_image2, alpha=0.03), cv2.COLORMAP_BONE)
			#combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_OPEN, kernel2)
			#combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_CLOSE, kernel2)
			#combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_OPEN, kernel3)
			#combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_CLOSE, kernel3)



			#ADAPTIVE THRESHOLDING
			#gray = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
			#gray2 = cv2.cvtColor(bg_removed,cv2.COLOR_BGR2GRAY)
			#thresh1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,6);
			#thresh2 = cv2.adaptiveThreshold(gray2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,6);
			#thresh3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,13);
			#thresh = np.asanyarray(thresh)
			#print(color_image)


			# Stack both images horizontally
			#images = np.hstack((color_image_blurred, depth_colormap,combined_colormap))
			#images2 = np.hstack((image_diff_colormap, image_diff_colormap2))

			# Show images
			cv2.imshow(window1Name, color_image)
			if ignoring_depth:
				cv2.imshow(window2Name,ignoring_image)
			else:
				cv2.imshow(window2Name,depth_colormap)
			cv2.imshow(window3Name,image_diff_colormap)
			cv2.imshow(window4Name,combined_colormap)
			cv2.imshow(window5Name,thresh_colormap)
			#cv2.imshow(window6Name,test_colormap)
			#cv2.imshow('thresh of unculled color image', thresh1)
			#cv2.imshow('thresh of culled color image', thresh2)
		#cv2.imshow('higher constant being subtracted', thresh3)
		if cv2.waitKey(33)==27:    # Esc key to stop
			break

finally:

    # Stop streaming
	pipeline.stop()
