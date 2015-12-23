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
			self.pMOG1 = cv2.createBackgroundSubtractorMOG2()
			self.pMOG2 = cv2.createBackgroundSubtractorMOG2()
			self.oldGray = None
			self.bottoms = []
			self.kp1 = None
			self.des1 = None
			self.colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
			
		# subsequent runs should empty out the MOG
		def clear(self):
			self.pMOG1 = cv2.createBackgroundSubtractorMOG2()
			self.pMOG2 = cv2.createBackgroundSubtractorMOG2()
			self.oldGray = None
			self.oldCorners = None
			self.kp1 = None
			self.des1 = None
			self.bottoms = []
			
		def findCenter(self, image, blurSigma, erodeElementSize, dilateElementSize, thresholdValue):
			# Gaussian blur
			blurred = cv2.GaussianBlur(image, (0,0), blurSigma)
			
			# Background subtraction
			foregroundMask = self.pMOG1.apply(image)
			
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
				
				return centroid, contours, thresholded, maxIndex
				
			else:
				return None, contours, thresholded, None


		# for each frame...
		def process(self, image, blurSigmaC, erodeElementSizeC, dilateElementSizeC, thresholdValueC, cannyHi, cannyLo, 
					blurSigma1, dilateElementSize1, erodeElementSize2, dilateElementSize2, thresholdValue, imageWidth, imageHeight):
			
			centroid, contours, contourCentering, largestContour = self.findCenter(image, blurSigmaC, erodeElementSizeC, dilateElementSizeC, thresholdValueC)
			
			contourCenteringColor = cv2.cvtColor(contourCentering, cv2.COLOR_GRAY2RGB)					
			if largestContour is not None:
				cv2.drawContours(contourCenteringColor, contours, largestContour, (0,255,0), 3)
				cv2.circle(contourCenteringColor, centroid, 8, (255,0,0), -1)
				
				
			# perform pyramid mean shift filtering to aid the thresholding step			
			#shifted = cv2.pyrMeanShiftFiltering(image, 21, 21)
			
			shifted = (cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(int)*255/180).astype(np.uint8)
			
			
			#shifted = cv2.bilateralFilter(image, 7, 150, 80)
			#shifted = np.multiply(shifted, (contourCentering!=0)[:,:,np.newaxis])
 
			# convert the mean shift image to grayscale, then apply Otsu's thresholding
			#gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
			gray = shifted[:,:,0]
			#thresh = contourCentering
			
			motionDetect = self.pMOG2.apply(image)

			erodeDilate = self.erodeDilate(motionDetect.copy(), erodeElementSize2, dilateElementSize2)
			
			#thresh = erodeDilate.copy()
			
			
			hueIsolate = np.multiply(gray, erodeDilate>120)	
			
			hueBins = 6
			binSize = int(255/hueBins)
			slices = 3
			
			hist = cv2.calcHist([hueIsolate],[0],erodeDilate,[hueBins],[0,256])
			
			print(hist[:,0])
			print([i[0] for i in sorted(enumerate(hist[:,0]), key=lambda x:x[1], reverse=True)])
			
			hueMasks = []
			
			for x in [i[0] for i in sorted(enumerate(hist[:,0]), key=lambda x:x[1], reverse=True)][:slices]:
				print(x*binSize, (x+1)*binSize)
				hueMasks.append(cv2.inRange(hueIsolate, x*binSize, (x+1)*binSize))
			
			colorIndex = 0
			
			segs = image.copy()

			for hueMask in hueMasks:
				thresh = np.multiply(hueMask, erodeDilate>120)				
				wcon = self.waterContour(thresh)
				print(self.colors[colorIndex])
				cv2.drawContours(segs, wcon, -1, self.colors[colorIndex])
				colorIndex += 1
			
			#thresh = 255-cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
			#thresh = erodeDilate
			
			
 
				# draw a circle enclosing the object
				#((x, y), r) = cv2.minEnclosingCircle(c)
				#cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
				#cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			
			# Gaussian blur
			#blurred = cv2.GaussianBlur(image, (0,0), blurSigma)
			
			#blurred = cv2.Laplacian(image, cv2.CV_8UC3)
			#blurred = cv2.threshold(blurred, thresholdValue, 1, cv2.THRESH_BINARY)[1]
			#blurred = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
			#canny = cv2.Canny(image, cannyLo, cannyHi)
			
			#erodeDilateCanny = cv2.multiply(self.erodeDilate(canny, 0, dilateElementSize1), contourCentering)
			##erodeDilateCanny = self.erodeDilate(canny, 0, dilateElementSize1)

			#_, contours2, hierarchy = cv2.findContours(erodeDilateCanny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			#erodeDilateCannyColor = cv2.cvtColor(erodeDilateCanny, cv2.COLOR_GRAY2RGB)		
				
			#numContours = 3
			#bigContourIdx = sorted([(cv2.contourArea(x), index) for index,x in enumerate(contours2)], reverse=True)[:numContours]
			
			
			
			
			#if len(contours2) > 0:
				
				#maxC = np.array(contours2[bigContourIdx[0][1]])
				
				##print(max(maxC[:,:,0])-min(maxC[:,:,0]),"\n")
				#cv2.rectangle(erodeDilateCannyColor, (min(maxC[:,:,0]),min(maxC[:,:,1])),(max(maxC[:,:,0]),max(maxC[:,:,1])), (0,0,255))
				#currentBottom = max(maxC[:,:,1])
				
				#bottoms = self.bottoms

				#bottoms.append(currentBottom)
				##print(bottoms)
				#if len(bottoms) > 5:
					
					#avgBottom = int(np.array(bottoms).mean())

					
					#if 0.95*avgBottom/imageHeight < currentBottom/float(imageHeight):
						#cv2.circle(erodeDilateCannyColor, (max(maxC[:,:,0]),max(maxC[:,:,1])), 10, (0,255,255), 5)
						#cv2.circle(erodeDilateCannyColor, (min(maxC[:,:,0]),max(maxC[:,:,1])), 10, (0,255,255), 5)
						
						
					#mask = np.zeros_like(erodeDilateCanny)
					#firstBL = (min(maxC[:,:,0])-20,min(imageHeight,avgBottom+20))
					#firstTR = (firstBL[0]+60, firstBL[1]-60)
					#secondBR = (max(maxC[:,:,0])+20,min(imageHeight,avgBottom+20))
					#secondTL = (secondBR[0]-60, secondBR[1]-60)
					
					#cv2.rectangle(mask, firstBL, firstTR, 255, -1)
					#cv2.rectangle(mask, secondBR, secondTL, 255, -1)
					
					## rows, cols
					#footMask1 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[avgBottom-40:min(imageHeight,avgBottom+20),min(maxC[:,:,0])-20:min(maxC[:,:,0])+40]
					#harris = cv2.cornerHarris(footMask1,2,3,0.04)
					
					#harrisPad = harris[:,:,np.newaxis]
					#harrisColor = np.dstack((harrisPad, np.zeros(harrisPad.shape), np.zeros(harrisPad.shape)))
					##image[avgBottom-30:min(imageHeight,avgBottom+30),min(maxC[:,:,0])-30:min(maxC[:,:,0])+30] = harrisColor
					
					##cv2.circle(image, (min(maxC[:,:,0]), avgBottom), 10, (0,255,0), 4)
					##cv2.circle(image, (max(maxC[:,:,0]), avgBottom), 10, (0,255,255), 4)
					
						
					#orb = cv2.ORB_create()

					#kp2, des2 = orb.detectAndCompute(erodeDilateCanny, mask)
					
					#if self.kp1 is not None and self.des1 is not None:
						## create BFMatcher object
						#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

						## Match descriptors.
						#matches = bf.match(self.des1,des2)

						## Sort them in the order of their distance.
						#matches = sorted(matches, key = lambda x:x.distance)
						
						#pointMatches = [(x, self.kp1[x.queryIdx], kp2[x.trainIdx]) for x in matches if self.tDist(self.kp1[x.queryIdx].pt, kp2[x.trainIdx].pt) < 40]
						
						

						## Draw first 10 matches.
						##for match in pointMatches[:10]:
					##		cv2.line(image, (int(match[1].pt[0]), int(match[1].pt[1])), (int(match[2].pt[0]), int(match[2].pt[1])), (255,0,255), 5)
						
					#self.kp1 = kp2.copy()
					#self.des1 = des2.copy()
						
						##erodeDilateCanny = cv2.drawKeypoints(erodeDilateCanny,kp2,color=(0,255,0), outImage=erodeDilateCanny)
				
			
			#contours3 = []
			#for contour in contours2:
				#contours3.append(cv2.convexHull(contour))
				
			#i = 0
			#for index in bigContourIdx:
				#hsv = self.hsv2RGBtuple(i, 255, 255)
				
				#cv2.drawContours(erodeDilateCannyColor, contours3, index[1], hsv, 2)
				#i += int(179.0/numContours)
				
			##cv2.drawContours(erodeDilateCannyColor, contours2, -1, (255,0,0), 2)
			

			## Background subtraction
			#motionDetect = self.pMOG2.apply(erodeDilateCanny)
			#blurMotionDetect = cv2.GaussianBlur(motionDetect, (0,0), blurSigma1)
			
			
			## Erode/Dilate:
			#dilated = self.erodeDilate(blurMotionDetect, erodeElementSize2, dilateElementSize2)

			## threshold - get rid of pixels with near-zero values
			#thresholded = cv2.threshold(dilated, thresholdValue, 255, cv2.THRESH_BINARY)[1]
			
			#if centroid is not None:
				#cannyCenter = self.reCenterX(canny, centroid)
				#erodeDilateCannyCenter = self.reCenterX(erodeDilateCanny, centroid)
				#erodeDilateCannyCenterColor = self.reCenterX(erodeDilateCannyColor, centroid)				
				#motionDetectCenter = self.reCenterX(motionDetect, centroid)
				#blurMotionDetectCenter = self.reCenterX(blurMotionDetect, centroid)
				#dilatedCenter = self.reCenterX(dilated, centroid)
				#threshholdedCenter = self.reCenterX(thresholded, centroid)
				#imageIsolate = self.reCenterX(np.multiply(thresholded[:,:,np.newaxis]/255, image), centroid)			
			#else:
				#cannyCenter = np.zeros_like(thresholded)
				#erodeDilateCannyCenter = np.zeros_like(thresholded)
				#erodeDilateCannyCenterColor = np.zeros_like(thresholded)
				#motionDetectCenter = np.zeros_like(thresholded)
				#blurMotionDetectCenter = np.zeros_like(thresholded)
				#dilatedCenter = np.zeros_like(thresholded)
				#threshholdedCenter = np.zeros_like(thresholded)
				#imageIsolate = np.zeros_like(thresholded)
			
			# find the contours (shapes) in the processed image
			#_, contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			
			# empty view to fill below
			#imageIsolate = np.zeros(image.shape, image.dtype)						
			
			# If we have contours, the biggest should be the person
			#if len(contours) > 0:
				## find the biggest contour
				#sizes = [cv2.contourArea(x) for x in contours]
				#maxIndex = sizes.index(max(sizes))
				
				## Outline the biggest contour on the morphed image
				#contourDisplay = cv2.cvtColor(thresholded*255, cv2.COLOR_GRAY2RGB)
				#cv2.drawContours(contourDisplay, contours, maxIndex, (0,255,0), 3)

				## Republish the original image, but only where there's motion
				#imageIsolate = np.zeros(image.shape, image.dtype)
				#cv2.drawContours(imageIsolate, contours, maxIndex, (1,1,1), -1)
				#imageIsolate = np.multiply(imageIsolate, image)
				
				## Find the center of mass of the largest contour
				#moments = cv2.moments(contours[maxIndex])
				#centroid = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))		
				
				#threshCenter = self.reCenterX(thresholded, centroid)
										
				## Recenter x-axis of the contour view and the republished motion view on the centroid
				#imageIsolate = self.reCenterX(imageIsolate, centroid)
				#contourDisplay = self.reCenterX(contourDisplay, centroid)
				#feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 2, blockSize = 7)
				
				
				##newGray = cv2.cvtColor(imageIsolate, cv2.COLOR_RGB2GRAY)
				##if self.oldGray is not None:
				
					##hsv = np.zeros_like(image)
					##hsv[...,1] = 255
					
					##flow = cv2.calcOpticalFlowFarneback(self.oldGray, newGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
					##mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
					##hsv[...,0] = ang*180/np.pi/2
					##hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
					##outFlow = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
				##else:
					##outFlow = newGray
				##self.oldGray = newGray.copy()
				
				##imageIsolate = cv2.cvtColor(imageIsolate, cv2.COLOR_RGB2GRAY)
				##showCorners = cv2.goodFeaturesToTrack(imageIsolate, mask = None, **feature_params)
				##print(showCorners)
				##for corner in showCorners:
				##	print(corner)
				##	cv2.circle(imageIsolate, (corner[0][0], corner[0][1]), 10, (255))
				
				
				##edgesCentered = self.reCenterX(edges, centroid)
				
				##doubleMOG = self.pMOG.apply(imageIsolate)
				##doubleMOGDilated = self.erodeDilate(doubleMOG, 4, 0)
				

			## If we didn't find a contour, just leave anything centered blank
			#else:
				#contourDisplay = np.zeros(image.shape, image.dtype)
				#threshCenter = thresholded
				#edgesCentered = edges
				#doubleMOGDilated = foregroundMask
			#	outFlow = image

			#newGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			#blackCanvas = np.zeros_like(self.oldGray)
			#if self.oldGray is not None:
				
				#hsv = np.zeros_like(image)
				#hsv[...,1] = 255
				
				#flow = cv2.calcOpticalFlowFarneback(self.oldGray, newGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
				#mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
				#hsv[...,0] = ang*180/np.pi/2
				#hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
				#outFlow = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
				
				#feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 2, blockSize = 7)

				#showCorners = cv2.goodFeaturesToTrack(newGray, mask = None, **feature_params)
				#print(showCorners)
				#for corner in showCorners:
					#print(corner)
					#cv2.circle(newGray, (corner[0][0], corner[0][1]), 10, (255))
				
				## Parameters for lucas kanade optical flow
				##lkparams = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
				##newCorners, st, err = cv2.calcOpticalFlowPyrLK(self.oldGray, newGray, self.oldCorners, None, **lkparams)				
				
				##print(len(self.oldCorners), len(newCorners))
				
				### Select good points
				##goodNew = newCorners[st==1]
				##goodOld = self.oldCorners[st==1]
				
				##print(goodNew, goodOld)
				
				### draw the tracks					
				##for new,old in zip(goodNew,goodOld):
					
				##	cv2.line(newGray, (new[0], new[1]), (old[0], old[1]), (0,255,255), 20)
				##self.oldCorners = newCorners.copy()
				
			#else:
				### params for ShiTomasi corner detection
				#feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
				#self.oldCorners = cv2.goodFeaturesToTrack(newGray, mask = None, **feature_params)
				##newGray = self.oldGray
				#outFlow = image
			
			#self.oldGray = newGray.copy()
			
			
			#outFlow = cv2.Sobel(image, cv2.CV_8UC1,1,0,ksize=5)#+cv2.Sobel(image, cv2.CV_8UC1,0,1,ksize=5)
			
			hi = self.showFour(hueMasks)
					
			return segs, hi, erodeDilate, motionDetect, hueIsolate
			#return imageIsolate, threshCenter, thresholded, foregroundMask, blurred
			
			
		def tDist(self, t1, t2):
			return math.hypot(t1[0]-t2[0], t1[1]-t2[1])
			
		def hsv2RGBtuple(self, h, s, v):
			return tuple(cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2RGB).astype(float)[0][0])
		
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
				
				if cv2.contourArea(c) < 400:
					continue
				
				epsilon = 0.02*cv2.arcLength(c,True)
				c = cv2.approxPolyDP(c, epsilon, True)
				
				outContours.append(c)
				
				#cv2.drawContours(segs, [c], -1, (0,255,255))
			
			return outContours
			
		def showFour(self, images):
			
			output = np.zeros_like(images[0])
			
			output[0:int(images[0].shape[0]/2),0:int(images[0].shape[1]/2)] = cv2.resize(images[0], (int(images[0].shape[1]/2), int(images[0].shape[0]/2)))
			
			if len(images) > 1:
				output[0:int(images[1].shape[0]/2),int(images[1].shape[1]/2):2*int(images[1].shape[1]/2)] = cv2.resize(images[1], (int(images[1].shape[1]/2), int(images[1].shape[0]/2)))
			
			if len(images) > 2:
				output[int(images[2].shape[0]/2):2*int(images[2].shape[0]/2),0:int(images[2].shape[1]/2)] = cv2.resize(images[2], (int(images[2].shape[1]/2), int(images[2].shape[0]/2)))
			
			if len(images) > 3:
				output[int(images[3].shape[0]/2):2*int(images[3].shape[0]/2),2*int(images[3].shape[1]/2):int(images[3].shape[1]/2)] = cv2.resize(images[3], (int(images[3].shape[1]/2), int(images[3].shape[0]/2)))
			
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
			
			
