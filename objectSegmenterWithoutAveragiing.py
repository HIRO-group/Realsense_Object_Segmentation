
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
import numpy as np
import cv2


# Configure depth and color streams. This currently sets autoexposure to ON.
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

#set exposure and other camera options
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
#depth_sensor.set_option(rs.option.gain, 16)
depth_sensor.set_option(rs.option.laser_power,360)
#depth_sensor.set_option(rs.option.frame_rate,30)


# Getting the depth sensor's depth scale
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

window1Name = 'LEFT: RGB image.  RIGHT: Depth subtracted image'


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
#cv2.namedWindow('thresh of unculled color image', cv2.WINDOW_AUTOSIZE)
#cv2.moveWindow('thresh of unculled color image',1500,20)
#cv2.namedWindow('thresh of culled color image', cv2.WINDOW_AUTOSIZE)
#cv2.moveWindow('thresh of culled color image',1500,620)
#cv2.namedWindow('higher constant being subtracted', cv2.WINDOW_AUTOSIZE)
#cv2.moveWindow('higher constant being subtracted',1500,1220)

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

		#if we haven't seen an image yet
#TODO: take average of first 10 frames
		if not firstFrameCaptured:
			depth_image = np.where(depth_image > 5, depth_image,0)
			firstDepthFrame = depth_image
			firstPicFrame = color_image
			firstFrameCaptured = True
		else:
			depth_image = np.where(depth_image > 5, depth_image,firstDepthFrame)


		if test: #this is just debugging info from the first frame
			print(depth_image[0])
			print(np.max(depth_image))
			#print(color_diffs[0][0])
			#print(depth_diffs[0])
			#print(culled_color_image[0][0])
			#print(len(color_image)==len(culled_color_image))
			test = False


		depth_diffs = cv2.subtract(firstDepthFrame,depth_image)
		color_diffs = cv2.subtract(firstPicFrame,color_image)



		#if there is a difference greater
		difference_threshold = 10 #the minimum difference in depth to care about
		white_color = 30000       #in COLORMAP_BONE, a really high number corresponds to white.
		black_color = 0           #in COLORMAP_BONE

		binary_depth_image = np.where(depth_diffs > difference_threshold,white_color,black_color)
		#culled_color_image = np.where(color_diffs > [30,30,30],[0,0,0],[200,200,200])


		# Remove background - Set pixels further than clipping_distance to grey
		#grey_color = 153
		#depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
		#bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0) | (depth_image_3d < clipping_distance_top), grey_color, color_image)



		# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(binary_depth_image, alpha=0.03), cv2.COLORMAP_BONE)



		#ADAPTIVE THRESHOLDING
		gray = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
		#gray2 = cv2.cvtColor(bg_removed,cv2.COLOR_BGR2GRAY)
		thresh1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,6);
		#thresh2 = cv2.adaptiveThreshold(gray2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,6);
		thresh3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,13);
		#thresh = np.asanyarray(thresh)
		#print(color_image)


		# Stack both images horizontally
		images = np.hstack((color_image, depth_colormap))

		# Show images
		cv2.imshow(window1Name, images)
		#cv2.imshow('thresh of unculled color image', thresh1)
		#cv2.imshow('thresh of culled color image', thresh2)
		#cv2.imshow('higher constant being subtracted', thresh3)
		if cv2.waitKey(33)==27:    # Esc key to stop
			break

finally:

    # Stop streaming
	pipeline.stop()
