
###################################################################
##      Intel Realsense image to binary segmentation image       ##
###################################################################

# AUTHOR : Jacob Fiola - HIRO robotics @ University of Colorado, Boulder
# LAST UPDATED - Mar 8 2020
# PY VERSION: 3.7.3
# DESCRIPTION: This is an algorithm with inputs = (rs.pipeline)
#	and output = 848x480 binary segmentation image

#	 pixels are determind to be part of an object by considering these factors:
# 	-FACTOR 1: Difference in depth for each pixel when compared to their base depths (The depth values acquired from calibration)
#   -FACTOR 2: Difference in R,G,and B for each pixel when compared to their base RGB values (The RGB values acquired from calibration)
#   -FACTOR 3: OpenCV's morphology tools to smooth out the segmentation


# algorithms to compare to 


do_mog1 = True

do_mog2 = True

do_lsbp = True

do_gsoc = True

do_unsupervised = True



# Import pyrealsense, numpy, and openCV
import pyrealsense2 as rs
import scipy.spatial as spp
import numpy as np
import cv2
import datetime
import argparse
import skimage.segmentation as seg
import skimage.color as color
import os
import colorsys

# Configure depth and color streams. This currently sets autoexposure to ON.
pipeline = rs.pipeline()
config = rs.config()
calibrating = True


# Dimension of output image
pixelHeight = 480
pixelWidth = 848


white_color = 30000       #in COLORMAP_BONE, a really high number corresponds to white.
black_color = 0           #in COLORMAP_BONE, this is black


# Calibration constants
numFramesToSkip = 20
numCalibrationFrames = 50


# Default thresholds before calibration (this doesnt matter)
color_difference_threshold_lighter_default = 30
color_difference_threshold_darker_default = 53
depth_constant_default = 12

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

# Size of kernels for pixel smoothing morphology and hole closing
kernel1 = np.ones((1,1),np.uint8)
kernel2 = np.ones((2,2),np.uint8)
kernel3 = np.ones((3,3),np.uint8)


# Choose which inputs to get from realsense
config.enable_stream(rs.stream.depth, pixelWidth, pixelHeight, rs.format.z16, 30)
config.enable_stream(rs.stream.color, pixelWidth, pixelHeight, rs.format.bgr8,15)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


# Start streaming
profile = pipeline.start(config)
device = profile.get_device()


# configure color sensor
color_sensor = device.query_sensors()[1]
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
color_sensor.set_option(rs.option.white_balance, 3700)
color_sensor.set_option(rs.option.exposure, 1200)
color_sensor.set_option(rs.option.gain, 0)
color_sensor.set_option(rs.option.gamma, 500)
#color_sensor.query_options()
color_sensor.set_option(rs.option.sharpness, 50)


# configure depth sensor
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
#depth_sensor.set_option(rs.option.gain, 16)
#depth_sensor.set_option(rs.option.exposure,10000)
depth_sensor.set_option(rs.option.laser_power,30)
#depth_sensor.set_option(rs.option.frame_rate,30)


# Getting the depth sensor's depth scale to convert depth values to metric
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)


#Window Names
rgbInputWinow = 'input image '
depthWindow = 'Depth difference'
rgbSubtractionWindow = 'RGB difference'
combinedSubtractionWindow = "RGB and Depth difference union"
watershedWindow = "Watershed"
mog1Window = "GMM based background subtraction"
mog2Window = "Improved GMM based background subtraction"
gmgWindow = "Background subtraction + Per-Pixel Bayesian segmentation"
lsbpWindow = "SVD binary pattern segmentation"
gsocWindow = "GSOC algotihm"
unsupervisedWindow = "My algorithm - unsupervised"

#region of interest stuff
fromCenter = False
roiChosen = False



# Recalibrate button callback
def recalibrate(newVal):
	framesCaptured = 0
	calibrating = True
	roiChosen = False

# Bound to some openCV trackbars which are simple value getter/setter trackbars
def empty(z):
	pass


# Create trackbars
cv2.namedWindow('controls',cv2.WINDOW_NORMAL)
cv2.resizeWindow("controls", 500,100);
cv2.createTrackbar('Snapshot', 'controls' , 0, 1, empty)
cv2.createTrackbar('RGB lighter difference threshold', 'controls' , 1, 255, empty)
cv2.createTrackbar('RGB darker difference threshold', 'controls' , 1, 255, empty)
cv2.createTrackbar('Recalibrate', 'controls' , 0, 1, empty)
cv2.createTrackbar('Depth Threshold', 'controls' , 1, 20, empty)

# Set trackbar default values
cv2.setTrackbarPos('RGB lighter difference threshold', 'controls', color_difference_threshold_lighter_default)
cv2.setTrackbarPos('RGB darker difference threshold', 'controls', color_difference_threshold_darker_default)
cv2.setTrackbarPos('Depth Threshold', 'controls', depth_constant_default)
cv2.setTrackbarPos('Snapshot','controls',0)

# Make windows
cv2.namedWindow(rgbInputWinow, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(depthWindow, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(rgbSubtractionWindow, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(combinedSubtractionWindow, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(unsupervisedWindow,cv2.WINDOW_AUTOSIZE)



xOffSet = pixelWidth +10
yOffSet = pixelHeight + 40

#if do_watershed:
#	cv2.namedWindow(watershedWindow, cv2.WINDOW_AUTOSIZE)
#	cv2.moveWindow(watershedWindow,0,2*yOffSet)

if do_mog1:
	mog1_fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG(history=500000)
	mog1_fgbg2 = cv2.bgsegm.createBackgroundSubtractorMOG(history=500000)
	#mog1_fgbg1.setNmixtures(1)
	#mog1_fgbg2.setNmixtures(1)
	cv2.namedWindow(mog1Window,cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow(mog1Window,0,2*yOffSet)

if do_mog2:
	mog2_fgbg1 = cv2.createBackgroundSubtractorMOG2(history = 500000,detectShadows=True)
	mog2_fgbg2 = cv2.createBackgroundSubtractorMOG2(history = 500000,detectShadows=True)
	cv2.namedWindow(mog2Window,cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow(mog2Window,0,2*yOffSet)


if do_lsbp:
	lsbp_fgbg1 = cv2.bgsegm.createBackgroundSubtractorLSBP()
	lsbp_fgbg2 = cv2.bgsegm.createBackgroundSubtractorLSBP()
	cv2.namedWindow(lsbpWindow,cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow(lsbpWindow,0,2*yOffSet)

if do_gsoc:
	gsoc_fgbg1 = cv2.bgsegm.createBackgroundSubtractorGSOC(hitsThreshold=500000)
	gsoc_fgbg2 = cv2.bgsegm.createBackgroundSubtractorGSOC(hitsThreshold=500000)
	cv2.namedWindow(gsocWindow,cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow(gsocWindow,0,2*yOffSet)

cv2.moveWindow(rgbInputWinow,0,0)
cv2.moveWindow(depthWindow,xOffSet,0)
cv2.moveWindow(rgbSubtractionWindow,0,yOffSet)
cv2.moveWindow(combinedSubtractionWindow,xOffSet,yOffSet)
cv2.moveWindow("controls",2*xOffSet,0)
cv2.moveWindow(unsupervisedWindow,0,2*yOffSet)


calibration_image = cv2.imread('calibrating.png',1)
ignoring_image = cv2.imread('ignoring.png',1)


framesCaptured = 0
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

		#select ROI
		if not roiChosen:
			region = cv2.selectROI(rgbInputWinow, np.asanyarray(color_frame.get_data()), fromCenter)
			start1 = region[1]
			end1 = region[1]+region[3]
			start2 = region[0]
			end2 = region[0] + region[2]
			if do_mog1:
				mog1_fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG(history=500)
				mog1_fgbg2 = cv2.bgsegm.createBackgroundSubtractorMOG(history=500)
			if do_mog2:
				mog2_fgbg1 = cv2.createBackgroundSubtractorMOG2(history = 500000,detectShadows=True)
				mog2_fgbg2 = cv2.createBackgroundSubtractorMOG2(history = 500000,detectShadows=True)
			if do_lsbp:
				lsbp_fgbg1 = cv2.bgsegm.createBackgroundSubtractorLSBP()
				lsbp_fgbg2 = cv2.bgsegm.createBackgroundSubtractorLSBP()
			if do_gsoc:
				gsoc_fgbg1 = cv2.bgsegm.createBackgroundSubtractorGSOC(hitsThreshold=500000)
				gsoc_fgbg2 = cv2.bgsegm.createBackgroundSubtractorGSOC(hitsThreshold=500000)
			print("region is " + str(region))
			roiChosen = True

		#get depth/color data and put into a numpy arrays
		depth_image = np.asanyarray(aligned_depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())

		#start1 = r[1]

		depth_image = depth_image[start1:end1, start2:end2]
		color_image = color_image[start1:end1, start2:end2]

		

		#color_image_blurred = cv2.blur(color_image,(3,3))
		b,g,r = cv2.split(color_image)
		calButton = cv2.getTrackbarPos('Recalibrate','controls')
		if calButton == 1:
			framesCaptured = 0
			calibrating = True
			roiChosen = False	
			cv2.setTrackbarPos('Recalibrate','controls',0)




		#------------------------------------------------ CALIBRATION ------------------------------------------------------------------------
		if(calibrating):

			#throw out some frames
			if framesCaptured < numFramesToSkip:
				framesCaptured +=1

			elif framesCaptured ==numFramesToSkip:
				cumulativeDepthFrame = depth_image.astype(np.uint32)
				numFramesNotZeroForPixel = np.where(depth_image > 0,1,0)
				bInitial,gInitial,rInitial = b,g,r
				framesCaptured +=1

			elif framesCaptured < numFramesToSkip+numCalibrationFrames:
				cumulativeDepthFrame = np.add(cumulativeDepthFrame.astype(np.uint32),depth_image.astype(np.uint32))
				bInitial = np.add(bInitial.astype(np.uint32),b.astype(np.uint32))
				gInitial = np.add(gInitial.astype(np.uint32),g.astype(np.uint32))
				rInitial = np.add(rInitial.astype(np.uint32),r.astype(np.uint32))
				numFramesNotZeroForPixel = np.where(depth_image > 0,numFramesNotZeroForPixel+1,numFramesNotZeroForPixel)
				framesCaptured +=1

			else:
				averageDepthFrame = np.divide(cumulativeDepthFrame,np.where(numFramesNotZeroForPixel>0,numFramesNotZeroForPixel,1))
				averageDepth = np.average(averageDepthFrame)
				averageDepthInMeters = depth_scale*averageDepth
				averageDepthInMM = averageDepthInMeters*1000
				bInitial = np.divide(bInitial,numCalibrationFrames).astype(np.uint8)
				gInitial = np.divide(gInitial,numCalibrationFrames).astype(np.uint8)
				rInitial = np.divide(rInitial,numCalibrationFrames).astype(np.uint8)
				averageR = np.average(rInitial)
				averageG = np.average(gInitial)
				averageB = np.average(bInitial)
				luminance = 0.2126*averageR + 0.7152*averageG + 0.0722*averageB


				h,s,v = rgb_to_hsv(averageR,averageG,averageB)


				color_difference_threshold_lighter_default = min(max([int(h*0.0183+s*3.751+v*-0.5178 + luminance*.2919-3.56),12]),65)
				color_difference_threshold_darker_default = min(max([int(h*-0.026+s*0.89+v*-0.4017 + luminance*.2572+2.33),13]),65)
				cv2.setTrackbarPos('RGB lighter difference threshold','controls',color_difference_threshold_lighter_default)
				cv2.setTrackbarPos('RGB darker difference threshold','controls',color_difference_threshold_darker_default)

				print("R: " + str(averageR)+ ", G: " + str(averageG) + ", B: " + str(averageB))
				print("H: " + str(h) + ", S: " + str(s) + "V: " + str(v))
				#print("Euclidean Length: " + str(euclidean_length))
				print("luminance: " + str(luminance))
				#print("RGB lighter threshold: " + str(color_difference_threshold_lighter_default))
				#print("RGB darker threshold: " + str(color_difference_threshold_darker_default))
				print("Average Depth: " + str(averageDepth))

				framesCaptured +=1

				if averageDepthInMeters > 2:
					depth_difference_threshold = 9999
					ignoring_depth = True
				else:
					depth_difference_threshold_default = int(averageDepth*.0264+0.0801)
					depth_difference_threshold = int(averageDepth*.0264+0.0801)
					cv2.setTrackbarPos('Depth Threshold','controls',depth_difference_threshold)
					print("depth difference threshhold " + str(depth_difference_threshold))
					ignoring_depth = False

				calibrating = False

			# Show images
			cv2.imshow(rgbInputWinow, calibration_image)
			cv2.imshow(depthWindow,calibration_image)
			cv2.imshow(rgbSubtractionWindow,calibration_image)
			cv2.imshow(combinedSubtractionWindow,calibration_image)


		#------------------------------------------------ SEGMENTATION ------------------------------------------------------------------------
		else:



			# Render the input window
			cv2.imshow(rgbInputWinow, color_image)

			#------------------------------------DEPTH SUBTRACTION------------------------------------------------------------------------

			# Get rid of bad depth values that failed to be computed for the current frame (shows up as black in RealSense Viewer)
			depth_image = np.where(depth_image > 0, depth_image,averageDepthFrame) 

			# Subtract the values of the average depth frame from the current depth frame
			depth_diffs = cv2.subtract(averageDepthFrame,depth_image)

			# Get the depth threshold
			depth_difference_threshold = cv2.getTrackbarPos('Depth Threshold', 'controls')

			# Assign white pixels in output image when
			binary_depth_image = np.where(depth_diffs > depth_difference_threshold,white_color,black_color)
			depth_colormap_culled = cv2.applyColorMap(cv2.convertScaleAbs(binary_depth_image, alpha=0.03), cv2.COLORMAP_BONE)

			binary_depth_image_unsupervised = np.where(depth_diffs > depth_difference_threshold_default,white_color,black_color)
			depth_colormap_culled_unsupervised = cv2.applyColorMap(cv2.convertScaleAbs(binary_depth_image_unsupervised, alpha=0.03), cv2.COLORMAP_BONE)

			# Smooth out the segmentation
			depth_colormap_culled = cv2.morphologyEx(depth_colormap_culled, cv2.MORPH_OPEN, kernel1)
			depth_colormap_culled = cv2.morphologyEx(depth_colormap_culled, cv2.MORPH_CLOSE, kernel1)

			depth_colormap_culled_unsupervised = cv2.morphologyEx(depth_colormap_culled_unsupervised, cv2.MORPH_OPEN, kernel1)
			depth_colormap_culled_unsupervised = cv2.morphologyEx(depth_colormap_culled_unsupervised, cv2.MORPH_OPEN, kernel1)

			# Render the depth subtraction window
			if ignoring_depth:
				cv2.imshow(depthWindow,ignoring_image)
			else:
				cv2.imshow(depthWindow,depth_colormap_culled)

			#------------------------------------RGB SUBTRACTION------------------------------------------------------------------------

			# get the threshold values from the trackbars. These values will be used in the R,G,and B difference thresholds.
			color_difference_threshold_darker = int(cv2.getTrackbarPos('RGB darker difference threshold', 'controls'))
			color_difference_threshold_lighter = int(cv2.getTrackbarPos('RGB lighter difference threshold', 'controls'))

			# Calculate the difference arrays for lighter and darker values in R,G and B.
			red_diffs_lighter = cv2.subtract(r,rInitial)
			red_diffs_darker = cv2.subtract(rInitial,r)
			green_diffs_lighter = cv2.subtract(g,gInitial)
			green_diffs_darker = cv2.subtract(gInitial,g)
			blue_diffs_lighter = cv2.subtract(b,bInitial)
			blue_diffs_darker = cv2.subtract(bInitial,b)

			# Assign white pixels in ouput image to when R, G, or B per-pixel differences are greater than lighter threshold values
			culled_color_image = np.where(blue_diffs_lighter > color_difference_threshold_lighter,white_color,black_color)
			culled_color_image = np.where(green_diffs_lighter > color_difference_threshold_lighter,white_color,culled_color_image)
			culled_color_image = np.where(red_diffs_lighter > color_difference_threshold_lighter,white_color,culled_color_image)

			culled_color_image_unsupervised = np.where(blue_diffs_lighter > color_difference_threshold_lighter_default,white_color,black_color)
			culled_color_image_unsupervised = np.where(green_diffs_lighter > color_difference_threshold_lighter_default,white_color,culled_color_image_unsupervised)
			culled_color_image_unsupervised = np.where(red_diffs_lighter > color_difference_threshold_lighter_default,white_color,culled_color_image_unsupervised)

			# Similarly, assign white pixels in output image to when R, G, or B per-pixel differences are greater than darker threshold values
			culled_color_image = np.where(blue_diffs_darker > color_difference_threshold_darker,white_color,culled_color_image)
			culled_color_image = np.where(green_diffs_darker > color_difference_threshold_darker,white_color,culled_color_image)
			culled_color_image = np.where(red_diffs_darker > color_difference_threshold_darker,white_color,culled_color_image)

			culled_color_image_unsupervised = np.where(blue_diffs_darker > color_difference_threshold_darker_default,white_color,culled_color_image_unsupervised)
			culled_color_image_unsupervised = np.where(green_diffs_darker > color_difference_threshold_darker_default,white_color,culled_color_image_unsupervised)
			culled_color_image_unsupervised = np.where(red_diffs_darker > color_difference_threshold_darker_default,white_color,culled_color_image_unsupervised)

			# Convert binary image to color map for display
			image_diff_colormap = cv2.applyColorMap(cv2.convertScaleAbs(culled_color_image, alpha=0.03), cv2.COLORMAP_BONE)
			image_diff_colormap_unsupervised = cv2.applyColorMap(cv2.convertScaleAbs(culled_color_image_unsupervised, alpha=0.03), cv2.COLORMAP_BONE)

			# Smooth out the image using morphology
			image_diff_colormap = cv2.morphologyEx(image_diff_colormap, cv2.MORPH_OPEN, kernel2)
			image_diff_colormap = cv2.morphologyEx(image_diff_colormap, cv2.MORPH_CLOSE, kernel2)
			image_diff_colormap = cv2.morphologyEx(image_diff_colormap, cv2.MORPH_OPEN, kernel1)
			image_diff_colormap = cv2.morphologyEx(image_diff_colormap, cv2.MORPH_CLOSE, kernel1)

			# Smooth out the image using morphology
			image_diff_colormap_unsupervised = cv2.morphologyEx(image_diff_colormap_unsupervised, cv2.MORPH_OPEN, kernel2)
			image_diff_colormap_unsupervised = cv2.morphologyEx(image_diff_colormap_unsupervised, cv2.MORPH_CLOSE, kernel2)
			image_diff_colormap_unsupervised = cv2.morphologyEx(image_diff_colormap_unsupervised, cv2.MORPH_OPEN, kernel1)
			image_diff_colormap_unsupervised = cv2.morphologyEx(image_diff_colormap_unsupervised, cv2.MORPH_CLOSE, kernel1)

			# Render the RGB subtraction window
			cv2.imshow(rgbSubtractionWindow,image_diff_colormap)

		#------------------------------------UNION OF DEPTH AND RGB SUBTRACTION--------------------------------------------------------

			combined = np.where(image_diff_colormap != 0, white_color,black_color)
			combined_unsupervised = np.where(image_diff_colormap_unsupervised != 0, white_color,black_color)
			if not ignoring_depth:
				combined = np.where(depth_colormap_culled != 0,white_color,combined)
				combined_unsupervised = np.where(depth_colormap_culled_unsupervised != 0,white_color,combined_unsupervised)

			combined_colormap = cv2.applyColorMap(cv2.convertScaleAbs(combined, alpha=0.03), cv2.COLORMAP_BONE)
			combined_colormap_unsupervised = cv2.applyColorMap(cv2.convertScaleAbs(combined_unsupervised, alpha=0.03), cv2.COLORMAP_BONE)
			
			#print("r[0] is " + str(r[0]))

			#combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_OPEN, kernel2)
			#combined_colormap = cv2.morphologyEx(combined_colormap, cv2.MORPH_CLOSE, kernel2)


			cv2.imshow(combinedSubtractionWindow,combined_colormap)

			if do_unsupervised:
				images5 = np.hstack((image_diff_colormap_unsupervised,depth_colormap_culled_unsupervised,combined_colormap_unsupervised))
				cv2.imshow(unsupervisedWindow,images5)



		#------------------------------------OTHER ALGORITHMS--------------------------------------------------------------------------
			#combined_diffs = np.add(blue_diffs,np.add(green_diffs,red_diffs))


		#WATERSHED
		#if do_watershed:
		#	gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
		#	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			#ret, thresh2 = cv2.threshold(blue_diffs_darker,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			#ret, thresh3 = cv2.threshold(green_diffs_lighter,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			#ret, thresh4 = cv2.threshold(green_diffs_darker,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			#ret, thresh5 = cv2.threshold(red_diffs_lighter,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			#ret, thresh6 = cv2.threshold(red_diffs_darker,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		#	thresh_image = np.where(thresh > 0, white_color, black_color)
		#	thresh_colormap = cv2.applyColorMap(cv2.convertScaleAbs(thresh_image, alpha=0.03), cv2.COLORMAP_BONE)
		#	cv2.imshow(watershedWindow,thresh_colormap)


		#GMM based- background subtraction

		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=.7), cv2.COLORMAP_HSV)

		if do_mog1:
			mog1fgmask1 = mog1_fgbg1.apply(color_image)
			mog1fgmask2 = mog1_fgbg2.apply(depth_colormap)
			mog1fgmask2 = np.where(depth_image > 0, mog1fgmask2,0)
			mog1fgmask3 = np.where(mog1fgmask1 >0, mog1fgmask1,mog1fgmask2)
			images1 = np.hstack((mog1fgmask1,mog1fgmask2,mog1fgmask3))
			cv2.imshow(mog1Window,images1)

		if do_mog2:
			mog2fgmask1 = mog2_fgbg1.apply(color_image)
			mog2fgmask1 = np.where(mog2fgmask1 >= 255, mog2fgmask1, 0)
			mog2fgmask2 = mog2_fgbg2.apply(depth_colormap)
			mog2fgmask2 = np.where(depth_image > 0, mog2fgmask2,0)
			mog2fgmask3 = np.where(mog2fgmask1 >0, mog2fgmask1,mog2fgmask2)
			images2 = np.hstack((mog2fgmask1,mog2fgmask2,mog2fgmask3))
			cv2.imshow(mog2Window,images2)

		if do_lsbp:
			lsbpfgmask1 = lsbp_fgbg1.apply(color_image)
			lsbpfgmask2 = lsbp_fgbg2.apply(depth_colormap)
			lsbpfgmask2 = np.where(depth_image > 0, lsbpfgmask2,0)
			lsbpfgmask3 = np.where(lsbpfgmask1 >0, lsbpfgmask1,lsbpfgmask2)
			images3 = np.hstack((lsbpfgmask1,lsbpfgmask2,lsbpfgmask3))
			cv2.imshow(lsbpWindow,images3)

		if do_gsoc:
			gsocfgmask1 = gsoc_fgbg1.apply(color_image)
			gsocfgmask2 = gsoc_fgbg2.apply(depth_colormap)
			gsocfgmask2 = np.where(depth_image > 0, gsocfgmask2,0)
			gsocfgmask3 = np.where(gsocfgmask1 >0, gsocfgmask1,gsocfgmask2)
			images4 = np.hstack((gsocfgmask1,gsocfgmask2,gsocfgmask3))
			cv2.imshow(gsocWindow,images4)



		snapshotButton = cv2.getTrackbarPos('Snapshot','controls')
		if snapshotButton == 1:
			os.chdir('C:/Users/Jacob/OneDrive/College/Senior Thesis/images/Performance Dataset') 
			now = str(datetime.datetime.now().strftime("%c").replace(":","-"))
			cv2.imwrite('ORIGINAL_COLOR_IMAGE_' + now+'.png', color_image)
			cv2.imwrite('ORIGINAL_DEPTH_COLORMAP_' + now+'.png', depth_colormap)
			cv2.imwrite('MOG1_COLOR_' + now+'.png', mog1fgmask1)
			cv2.imwrite('MOG1_DEPTH_' + now+'.png', mog1fgmask2)
			cv2.imwrite('MOG1_COMBINED_' + now+'.png', mog1fgmask3)
			cv2.imwrite('MOG2_COLOR_' + now+'.png', mog2fgmask1)
			cv2.imwrite('MOG2_DEPTH_' + now+'.png', mog2fgmask2)
			cv2.imwrite('MOG2_COMBINED_' + now+'.png', mog2fgmask3)
			cv2.imwrite('LSBP_COLOR_' + now+'.png', lsbpfgmask1)
			cv2.imwrite('LSBP_DEPTH_' + now+'.png', lsbpfgmask2)
			cv2.imwrite('LSBP_COMBINED_' + now+'.png', lsbpfgmask3)
			cv2.imwrite('GSOC_COLOR_' + now+'.png', gsocfgmask1)
			cv2.imwrite('GSOC_DEPTH_' + now+'.png', gsocfgmask2)
			cv2.imwrite('GSOC_COMBINED_' + now+'.png', gsocfgmask3)
			cv2.imwrite('MY_COLOR_SUPERVISED' + now+'.png', image_diff_colormap)
			cv2.imwrite('MY_DEPTH_SUPERVISED' + now+'.png', depth_colormap_culled)
			cv2.imwrite('MY_COMBINED_SUPERVISED' + now+'.png', combined_colormap)
			cv2.imwrite('MY_COLOR_UNSUPERVISED' + now+'.png', image_diff_colormap_unsupervised)
			cv2.imwrite('MY_DEPTH_UNSUPERVISED' + now+'.png', depth_colormap_culled_unsupervised)
			cv2.imwrite('MY_COMBINED_UNSUPERVISED' + now+'.png', combined_colormap_unsupervised)
			cv2.setTrackbarPos('Snapshot','controls',0)

		if cv2.waitKey(33)==27:    # Esc key to stop
			break
finally:
    # Stop streaming
	pipeline.stop()
	cv2.destroyAllWindows()
