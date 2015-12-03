import cv2

class PointLightWalker():
	
		def __init__(self):			
			# Setup the Background Subtractor
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
						
			return thresholded, dilated, eroded, foregroundMask, blurred
