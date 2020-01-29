
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


# Configure depth and color streams. This currently sets autoexposure to ON.
pipeline = rs.pipeline()
config = rs.config()
calibrating = True

pixelHeight = 480
pixelWidth = 848

numCalibrationFrames = 30



kernel = np.ones((3,3),np.uint8)
kernel2 = np.ones((8,8),np.uint8)
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

color_difference_threshold_darker = 150

alpha_slider_max = 100
title_window = 'RGB Darker threshold'

#def on_trackbar(val):
#	alpha = val / alpha_slider_max
#	beta = ( 1.0 - alpha )
#	color_difference_threshold_darker = alpha*200
#	print(color_difference_threshold_darker)

#trackbar_name = 'Alpha x %d' % alpha_slider_max
#cv2.namedWindow('test')
#cv2.createTrackbar(trackbar_name, 'test' , 0, alpha_slider_max, on_trackbar)

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

cv2.namedWindow(window2Name, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(window2Name,0,800)

cv2.namedWindow(window3Name, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(window3Name,1000,20)

cv2.namedWindow(window4Name, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(window4Name,1000,800)

#cv2.namedWindow('thresh of unculled color image', cv2.WINDOW_AUTOSIZE)
#cv2.moveWindow('thresh of unculled color image',1500,20)
#cv2.namedWindow('thresh of culled color image', cv2.WINDOW_AUTOSIZE)
#cv2.moveWindow('thresh of culled color image',1500,620)
#cv2.namedWindow('higher constant being subtracted', cv2.WINDOW_AUTOSIZE)
#cv2.moveWindow('higher constant being subtracted',1500,1220)


calibration_image = cv2.imread('calibrating.png',1)


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
		#color_image_blurred = cv2.blur(color_image,(3,3))
		b,g,r = cv2.split(color_image)




		#if we haven't seen an image yet
		if(calibrating):
			if framesCaptured <20:
				framesCaptured +=1
			elif framesCaptured ==20:
				firstDepthFrame = depth_image.astype(np.uint32)
				numFramesNotZeroForPixel = np.where(depth_image > 0,1,0)
				bInitial,gInitial,rInitial = b,g,r
				framesCaptured +=1
			elif framesCaptured < 40:
				firstDepthFrame = np.add(firstDepthFrame.astype(np.uint32),depth_image.astype(np.uint32))
				bInitial = np.add(bInitial.astype(np.uint32),b.astype(np.uint32))
				gInitial = np.add(gInitial.astype(np.uint32),g.astype(np.uint32))
				rInitial = np.add(rInitial.astype(np.uint32),r.astype(np.uint32))
				numFramesNotZeroForPixel = np.where(depth_image > 0,numFramesNotZeroForPixel+1,numFramesNotZeroForPixel)
				framesCaptured +=1
			else:
				averageDepthFrame = np.divide(firstDepthFrame,np.where(numFramesNotZeroForPixel>0,numFramesNotZeroForPixel,1))
				#print("b Initial before division",bInitial[0][0])
				bInitial = np.divide(bInitial,20).astype(np.uint8)
				gInitial = np.divide(gInitial,20).astype(np.uint8)
				rInitial = np.divide(rInitial,20).astype(np.uint8)
				#print("b Initial after division",bInitial[0][0])
				#initialColors = np.sqrt(np.add(np.multiply(bInitial,bInitial),np.add(np.multiply(gInitial,gInitial),np.multiply(rInitial,rInitial)))).astype(np.uint8)
				framesCaptured +=1
				calibrating = False



				# Show images
			cv2.imshow(window1Name, calibration_image)

		else:
			depth_image = np.where(depth_image > 0, depth_image,averageDepthFrame)
			depth_diffs = cv2.subtract(averageDepthFrame,depth_image)

			#currColors = np.sqrt(np.add(np.multiply(b,b),np.add(np.multiply(g,g),np.multiply(r,r)))).astype(np.uint8)
			#color_diffs = np.sqrt(np.add(np.square(np.absolute(np.subtract(b,bInitial))),np.add(np.square(np.absolute(np.subtract(g,gInitial))),np.square(np.absolute(np.subtract(r,rInitial))))))
			#print(color_diffs[0][0])

			blue_diffs_lighter = cv2.subtract(b,bInitial)
			blue_diffs_darker = cv2.subtract(bInitial,b)
			green_diffs_lighter = cv2.subtract(g,gInitial)
			green_diffs_darker = cv2.subtract(gInitial,g)
			red_diffs_lighter = cv2.subtract(r,rInitial)
			red_diffs_darker = cv2.subtract(rInitial,r)

			#blue_diffs = cv2.absdiff(b,bInitial)
			#green_diffs = cv2.absdiff(g,gInitial)
			#red_diffs = cv2.absdiff(r,rInitial)


			if test: #this is just debugging info from the first frame
				#print(depth_image[0])
				#print(len(depth_image[0]))
				#print(np.max(depth_image))
				#print(green_diffs[0][0])
				#print(depth_diffs[0])
				#print(culled_color_image[0][0])
				#print(len(color_image)==len(culled_color_image))
				test = False

			#if there is a difference greater
			depth_difference_threshold = 10 #the minimum difference in depth to care about
			color_difference_threshold_lighter = 15
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

			#print("binitial",bInitial[0][0])
			#print("bcurr",b[0][0])

			culled_color_image = np.where(blue_diffs_darker > color_difference_threshold_darker,white_color,culled_color_image)
			culled_color_image = np.where(green_diffs_darker > color_difference_threshold_darker,white_color,culled_color_image)
			culled_color_image = np.where(red_diffs_darker > color_difference_threshold_darker,white_color,culled_color_image)
			#culled_color_image = cv2.blur(culled_color_image,(10,10))




			#print(red_diffs_darker[0][0])

			#culled_color_image = np.where(blue_diffs > color_difference_threshold_test1,white_color,black_color)
			#culled_color_image = np.where(green_diffs > color_difference_threshold_test1,white_color,culled_color_image)
			#culled_color_image = np.where(red_diffs > color_difference_threshold_test1,white_color,culled_color_image)
			#culled_color_image = np.where(blue_diffs > color_difference_threshold_test,black_color,culled_color_image)
			#culled_color_image = np.where(green_diffs > color_difference_threshold_test,black_color,culled_color_image)
			#culled_color_image = np.where(red_diffs > color_difference_threshold_test,black_color,culled_color_image)

			#culled_color_image = np.where(color_diffs > color_difference_threshold_test2,white_color,black_color)

			combined_image = np.where(culled_color_image == white_color,white_color,black_color)
			combined_image = np.where(binary_depth_image == white_color,white_color,combined_image)

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
			depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(binary_depth_image, alpha=0.03), cv2.COLORMAP_BONE)
			image_diff_colormap = cv2.applyColorMap(cv2.convertScaleAbs(culled_color_image, alpha=0.03), cv2.COLORMAP_BONE)
			combined_colormap = cv2.applyColorMap(cv2.convertScaleAbs(combined_image, alpha=0.03), cv2.COLORMAP_BONE)

			depth_colormap = cv2.morphologyEx(depth_colormap, cv2.MORPH_OPEN, kernel)
			depth_colormap= cv2.morphologyEx(depth_colormap, cv2.MORPH_CLOSE, kernel)

			#image_diff_colormap = cv2.morphologyEx(image_diff_colormap, cv2.MORPH_OPEN, kernel2)
			#image_diff_colormap = cv2.morphologyEx(image_diff_colormap, cv2.MORPH_OPEN, kernel2)
			#image_diff_colormap2= cv2.applyColorMap(cv2.convertScaleAbs(culled_color_image2, alpha=0.03), cv2.COLORMAP_BONE)
			combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_OPEN, kernel2)
			combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_CLOSE, kernel2)
			combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_OPEN, kernel3)
			combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_CLOSE, kernel3)



			#ADAPTIVE THRESHOLDING
			gray = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
			#gray2 = cv2.cvtColor(bg_removed,cv2.COLOR_BGR2GRAY)
			thresh1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,6);
			#thresh2 = cv2.adaptiveThreshold(gray2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,6);
			thresh3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,13);
			#thresh = np.asanyarray(thresh)
			#print(color_image)


			# Stack both images horizontally
			#images = np.hstack((color_image_blurred, depth_colormap,combined_colormap))
			#images2 = np.hstack((image_diff_colormap, image_diff_colormap2))

			# Show images
			cv2.imshow(window1Name, color_image)
			cv2.imshow(window2Name,depth_colormap)
			cv2.imshow(window3Name,image_diff_colormap)
			cv2.imshow(window4Name,combined_colormap)
			#cv2.imshow('thresh of unculled color image', thresh1)
			#cv2.imshow('thresh of culled color image', thresh2)
		#cv2.imshow('higher constant being subtracted', thresh3)
		if cv2.waitKey(33)==27:    # Esc key to stop
			break

finally:

    # Stop streaming
	pipeline.stop()
