import numpy as np
import cv2

class PointLightWalker():
	
		def __init__(self):			
			# Setup the Background Subtractor
			self.pMOG = cv2.createBackgroundSubtractorMOG2()
			
		# subsequent runs should empty out the MOG
		def clear(self):
			self.pMOG = cv2.createBackgroundSubtractorMOG2()

		def process(self, image, blurSigma, erodeElementSize, dilateElementSize, thresholdValue):
						
			# Gaussian blur
			blurred = cv2.GaussianBlur(image, (0,0), blurSigma)

			# Background subtraction
			foregroundMask = self.pMOG.apply(blurred)
			
			# Erode/Dilate:
			if erodeElementSize > 0:
				self.erodeElement = cv2.getStructuringElement(cv2.MORPH_RECT, (erodeElementSize,erodeElementSize))		
				eroded = cv2.erode(foregroundMask, self.erodeElement) 
			else:
				eroded = foregroundMask
				
			if dilateElementSize > 0:
				self.dilateElement = cv2.getStructuringElement(cv2.MORPH_RECT, (dilateElementSize,dilateElementSize))
				dilated = cv2.dilate(eroded, self.dilateElement)          
			else:
				dilated = eroded

			# threshold - get rid of pixels with near-zero values
			thresholded = cv2.threshold(dilated, thresholdValue, 255, cv2.THRESH_BINARY)[1]
			
			# find the contours (shapes) in the processed image
			_, contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			
			# empty view to fill below
			imageIsolate = np.zeros(image.shape, image.dtype)
			
			# If we have contours, the biggest should be the person
			if len(contours) > 0:
				# find the biggest contour
				sizes = [cv2.contourArea(x) for x in contours]
				maxIndex = sizes.index(max(sizes))
				
				# Outline the biggest contour on the morphed image
				contourDisplay = cv2.cvtColor(thresholded*255, cv2.COLOR_GRAY2RGB)
				cv2.drawContours(contourDisplay, contours, maxIndex, (0,255,0), 3)

				# Republish the original image, but only where there's motion
				imageIsolate = np.zeros(image.shape, image.dtype)
				cv2.drawContours(imageIsolate, contours, maxIndex, (1,1,1), -1)
				imageIsolate = np.multiply(imageIsolate, image)
				
				# Find the center of mass of the largest contour
				moments = cv2.moments(contours[maxIndex])
				centroid = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))		
										
				# Recenter x-axis of the contour view and the republished motion view on the centroid
				imageIsolate = self.reCenterX(imageIsolate, centroid)
				contourDisplay = self.reCenterX(contourDisplay, centroid)

			# If we didn't find a contour, just leave anything centered blank
			else:
				contourDisplay = np.zeros(image.shape, image.dtype)
						
			return imageIsolate, contourDisplay, dilated, foregroundMask, blurred

		def reCenter(self, image, point, displayCentroid = True):
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
			
		def reCenterY(self, image, point, displayCentroid = True):
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
			
		def reCenterX(self, image, point, displayCentroid = True):
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
			
			
