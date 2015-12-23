import numpy as np
import cv2
import math
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

# Do the PointLightWalker processing
class PointLightWalker():
	
		def __init__(self):			
			# Setup the Background Subtractor 
			self.pMOG = cv2.createBackgroundSubtractorMOG2()
			# handy color array for contours
			self.colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
			
		# subsequent runs should empty out the MOG
		def clear(self):
			self.pMOG = cv2.createBackgroundSubtractorMOG2()			
			
		def findCenter(self, image, blurSigma, erodeElementSize, dilateElementSize, thresholdValue):
			# Gaussian blur
			blurred = cv2.GaussianBlur(image, (0,0), blurSigma)
			
			# Background subtraction
			foregroundMask = self.pMOG.apply(image)
			
			# Erode/Dilate:
			dilated = self.erodeDilate(foregroundMask, erodeElementSize, dilateElementSize)
			
			# threshold - get rid of pixels with near-zero values
			thresholded = cv2.threshold(dilated, thresholdValue, 255, cv2.THRESH_BINARY)[1]
			
			# find the contours (shapes) in the processed image
			_, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				
			
			# If we have contours, the biggest should be the person
			if len(contours) > 0:
				# find the biggest contour
				sizes = [cv2.contourArea(x) for x in contours]
				maxIndex = sizes.index(max(sizes))
				
				# Find the center of mass of the largest contour
				moments = cv2.moments(contours[maxIndex])
				centroid = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))		
				
				return centroid, contours, thresholded, maxIndex, foregroundMask, thresholded
				
			else:
				return None, contours, thresholded, None, foregroundMask, thresholded


		# for each frame...
		def process(self, image, blurSigmaC, erodeElementSizeC, dilateElementSizeC, thresholdValueC, hueBins, slices, watershedDistance, imageWidth, imageHeight, dumpFirstHue):
			
			# Run the motion detection and find the center of mass of the walker
			centroid, contours, contourCentering, largestContour, motionDetect, erodeDilate = self.findCenter(image, blurSigmaC, erodeElementSizeC, dilateElementSizeC, thresholdValueC)
						
			# Morph the detect motion to reduce noise and beef it up a little
			#erodeDilate = self.erodeDilate(motionDetect.copy(), erodeElementSize2, dilateElementSize2)

			# Create a grayscale image based on hue level
			shifted = (cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(int)*255/180).astype(np.uint8)
			gray = shifted[:,:,0]
			
			# Isolate the hue to motion detected regions only
			hueIsolate = np.multiply(gray, erodeDilate>120)	
			
			# How many pixels within the motion detected area fall within each hue range?
			binSize = int(255/hueBins)
			hist = cv2.calcHist([hueIsolate],[0],erodeDilate,[hueBins],[0,256])
			
			# We are going to have a binary mask for the location of each hue range within the motion detected area			
			hueMasks = []
			for x in [i[0] for i in sorted(enumerate(hist[:,0]), key=lambda x:x[1], reverse=True)][dumpFirstHue:slices]:
				hueMasks.append(cv2.inRange(hueIsolate, x*binSize, (x+1)*binSize))
			
			# Also track the walker shape as a whole
			hueMasks.append(cv2.threshold(erodeDilate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
			
			# Write each set of hue motion segmentation contours to an output display (as a distinct color - limit 6)
			colorIndex = 0
			segs = np.zeros_like(image)			
			for hueMask in hueMasks:
				# Isolate motion detected area
				thresh = np.multiply(hueMask, erodeDilate>120)				
				# Run watershed contouring
				wcon = self.waterContour(thresh)
				cv2.drawContours(segs, wcon, -1, self.colors[colorIndex])
				colorIndex += 1
			
			# show all masks at once
			allMasks = self.showFour(hueMasks)
			
			# Center the final result
			if centroid is not None:
				segs = self.reCenterX(segs, centroid)
					
			return segs, allMasks, erodeDilate, motionDetect, hueIsolate
			
		# Watershed image segmentation algorithm
		# see http://docs.opencv.org/master/d2/dbd/tutorial_distance_transform.html#gsc.tab=0
		def waterContour(self, thresh):
			
			# compute the exact Euclidean distance from every binary pixel to the nearest zero pixel, then find peaks in this distance map
			D = ndimage.distance_transform_edt(thresh)
			localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
 
			# perform a connected component analysis on the local peaks, using 8-connectivity, then appy the Watershed algorithm
			markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
			labels = watershed(-D, markers, mask=thresh)
			
			
			outContours = []
			
			# loop over the unique labels returned by the Watershed algorithm
			for label in np.unique(labels):
				# if the label is zero, we are examining the 'background' so simply ignore it
				if label == 0:
					continue
 
				# otherwise, allocate memory for the label region and draw it on the mask
				mask = np.zeros(thresh.shape, dtype="uint8")
				mask[labels == label] = 255
 
				# detect contours in the mask and grab the largest one
				cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
				c = max(cnts, key=cv2.contourArea)
				
				# Get rid of small contours
				if cv2.contourArea(c) < 400:
					continue
				
				epsilon = 0.02*cv2.arcLength(c,True)
				c = cv2.approxPolyDP(c, epsilon, True)
				
				outContours.append(c)
			
			return outContours
			
		# Display between one and four images (halved) within the same display
		def showFour(self, images):
			
			output = np.zeros_like(images[0])
			
			output[0:int(images[0].shape[0]/2),0:int(images[0].shape[1]/2)] = cv2.resize(images[0], (int(images[0].shape[1]/2), int(images[0].shape[0]/2)))
			
			if len(images) > 1:
				output[0:int(images[1].shape[0]/2),int(images[1].shape[1]/2):2*int(images[1].shape[1]/2)] = cv2.resize(images[1], (int(images[1].shape[1]/2), int(images[1].shape[0]/2)))
			
			if len(images) > 2:
				output[int(images[2].shape[0]/2):2*int(images[2].shape[0]/2),0:int(images[2].shape[1]/2)] = cv2.resize(images[2], (int(images[2].shape[1]/2), int(images[2].shape[0]/2)))
			
			if len(images) > 3:
				output[int(images[3].shape[0]/2):2*int(images[3].shape[0]/2),int(images[3].shape[1]/2):2*int(images[3].shape[1]/2)] = cv2.resize(images[3], (int(images[3].shape[1]/2), int(images[3].shape[0]/2)))
			
			return output
			
				
		# Center the input image on the given point.  Zeros everywhere new
		def reCenter(self, image, point, displayCentroid = False):
			reCentered = np.zeros(image.shape, image.dtype)
			
			# remember, point is (x,y) and shape is (rows, cols) or (y,x)
			left = point[0] - image.shape[1]/2
			right = point[0] + image.shape[1]/2
			top = point[1] - image.shape[0]/2
			bottom = point[1] + image.shape[0]/2
						
			# Where to slice from in the old image (y,x)? Saturate
			leftOld = max(0, left)
			rightOld = min(reCentered.shape[1], right)			
			topOld = max(0, top)			
			bottomOld = min(reCentered.shape[0], bottom)
						
			leftNew = max(0, -left)
			topNew = max(0, -top)
			rightNew = leftNew + (rightOld - leftOld)
			bottomNew = topNew + (bottomOld - topOld)
			
			# slice the still-visible image portion into reCentered:
			reCentered[topNew:bottomNew, leftNew:rightNew] = image[topOld:bottomOld,leftOld:rightOld]
			
			# If requested, display the centroid
			if displayCentroid:
				cv2.circle(reCentered, (int(image.shape[1]/2), int(image.shape[0]/2)), 8, (255,0,0), -1)
			
			return reCentered
			
		def erodeDilate(self, image, erodeElementSize, dilateElementSize):
			if erodeElementSize > 0:
				self.erodeElement = cv2.getStructuringElement(cv2.MORPH_RECT, (erodeElementSize,erodeElementSize))		
				eroded = cv2.erode(image, self.erodeElement) 
			else:
				eroded = image
				
			if dilateElementSize > 0:
				self.dilateElement = cv2.getStructuringElement(cv2.MORPH_RECT, (dilateElementSize,dilateElementSize))
				dilated = cv2.dilate(eroded, self.dilateElement)          
			else:
				dilated = eroded
				
			return dilated
				
		# Center the input image's height on the given point's Y coordinate.  Zeros everywhere new
		def reCenterY(self, image, point, displayCentroid = False):
			reCentered = np.zeros(image.shape, image.dtype)
			
			# remember, point is (x,y) and shape is (rows, cols) or (y,x)
			top = point[1] - image.shape[0]/2
			bottom = point[1] + image.shape[0]/2
						
			# Where to slice from in the old image (y,x)? Saturate		
			topOld = max(0, top)			
			bottomOld = min(reCentered.shape[0], bottom)
						
			topNew = max(0, -top)
			bottomNew = topNew + (bottomOld - topOld)
			
			# slice the still-visible image portion into reCentered:
			reCentered[topNew:bottomNew, :] = image[topOld:bottomOld,:]
			
			# If requested, display the centroid
			if displayCentroid:
				cv2.circle(reCentered, (int(point[0]), int(image.shape[0]/2)), 8, (255,0,0), -1)
			
			return reCentered
			
		# Center the input image's width on the given point's X coordinate.  Zeros everywhere new
		def reCenterX(self, image, point, displayCentroid = False):
			reCentered = np.zeros(image.shape, image.dtype)
			
			# remember, point is (x,y) and shape is (rows, cols) or (y,x)
			left = point[0] - image.shape[1]/2
			right = point[0] + image.shape[1]/2
						
			# Where to slice from in the old image (y,x)? Saturate
			leftOld = max(0, left)
			rightOld = min(reCentered.shape[1], right)			
						
			leftNew = max(0, -left)
			rightNew = leftNew + (rightOld - leftOld)
			
			# slice the still-visible image portion into reCentered:
			reCentered[:, leftNew:rightNew] = image[:,leftOld:rightOld]
			
			# If requested, display the centroid
			if displayCentroid:
				cv2.circle(reCentered, (int(image.shape[1]/2), int(point[1])), 8, (255,0,0), -1)
			
			return reCentered
			
			
