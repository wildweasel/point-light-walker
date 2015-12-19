# Matt Kaplan, CISC 642 Assigment 2, Fall '15
import cv2
from tkinter import *
import PIL
from PIL import Image, ImageTk
import numpy as np

# Add numpy image array viewing (with Scaling) to a basic Tkinter canvas
class OpenCVCanvas(Canvas):

	def __init__(self, args, **kwargs):
		super().__init__(args, kwargs)
		self.img = None
		
		# Set up acceptible save image file types
		self.saveFileOpt = {}
		self.saveFileOpt['filetypes'] = [('JPEG images', '.jpg'),('PNG images', '.png'),('PPM images', '.ppm')]

	# Display numpy array as image on GUI			
	def publishArray(self, numpyArray):
		
			# keep a copy of the raw numpy image
			self.img = numpyArray
		
			self.shrinkRatio = 1.0
		
			# Scale to fit window size
			if numpyArray.shape[0] > self.winfo_height() or numpyArray.shape[1] > self.winfo_width():
				
				# rows/cols  , y/x
				arrayRatio = float(numpyArray.shape[0])/numpyArray.shape[1]						
				canvasRatio = float(self.winfo_height() / self.winfo_width())
				
				# array height controls
				if arrayRatio > canvasRatio:
					self.shrinkRatio = float(self.winfo_height()) / numpyArray.shape[0]
				# array width controls
				else:
					self.shrinkRatio = float(self.winfo_width()) / numpyArray.shape[1]
								
				numpyArray = cv2.resize(numpyArray, (0,0), fx=self.shrinkRatio, fy=self.shrinkRatio)				
				
			# Turn the numpy array into an image
			self.image = Image.fromarray(numpyArray.astype(np.uint8))

			# display on the canvas
			self.imagePhoto = ImageTk.PhotoImage(self.image)
			self.create_image((numpyArray.shape[1]/2,numpyArray.shape[0]/2),image=self.imagePhoto)
		
	# query user and open image
	def loadImage(self):
		
		openImage = filedialog.askopenfilename()
		
		if openImage:			
			image = cv2.imread(openImage)
			if image is not None:
				# OpenCV reads in BGR - switch to RGB
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				self.publishArray(image)
				return True
			else:
				return False
		else:
			return False
			
	# Query user and save image
	def saveImage(self):
		
		saveImage = filedialog.asksaveasfilename(**self.saveFileOpt)
		
		if saveImage and self.img is not None:
			# OpenCV writes in BGR - switch from RGB
			out = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
			cv2.imwrite(saveImage, out)
			
	# Display an image pyramid in the canvas
	def publishPyramid(self, pyramid):
		self.publishArray(self.FlattenPyramid(pyramid))
			
	# squash a pyramid (list of opencv images) for easy display
	# The pyramid must come in as a list of opencv/numpy images in descending size.  
	# This function is generally designed for a pyramid that scales down by 2 at each image, 
	#	but should also be able to handle pyramids that scale down faster 
	#	(or any descending pyramid in which the height of the first image is greater than the sum of the heights of all the other images)
	def FlattenPyramid(self, pyramid):
				
		# If we've only got one image in the list, there's nothing to flatten.  (Also prevents empty-list errors)
		if len(pyramid) < 2:
			return pyramid[0]
		
		# Blank black canvas
		output = np.zeros((pyramid[0].shape[0], pyramid[0].shape[1]+pyramid[1].shape[1], pyramid[0].shape[2]), dtype=np.uint8)
		
		# Draw the first image on the left
		output[0:pyramid[0].shape[0], 0:pyramid[0].shape[1]] = pyramid[0]
		
		# Draw each sucessive image on the right, starting at the top of the canvas and cascading down
		startY = 0
		for y in range(1, len(pyramid)):
			output[startY:startY+pyramid[y].shape[0], pyramid[0].shape[1]:pyramid[0].shape[1]+pyramid[y].shape[1]] = pyramid[y]
			startY += pyramid[y].shape[0]

		return output

