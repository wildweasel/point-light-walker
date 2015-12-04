import cv2
from tkinter import *
import PIL
from PIL import Image, ImageTk
import numpy as np

# Add numpy image array viewing (with Scaling) to a basic Tkinter canvas
class OpenCVCanvas(Canvas):

	# Display numpy array as image on GUI			
	def publishArray(self, numpyArray):
		
			# keep a copy of the raw numpy image
			self.img = numpyArray
		
			# Scale to fit window size
			if numpyArray.shape[0] > self.winfo_height() or numpyArray.shape[1] > self.winfo_width():
				
				# rows/cols  , y/x
				arrayRatio = float(numpyArray.shape[0])/numpyArray.shape[0]						
				canvasRatio = float(self.winfo_height() / self.winfo_width())
				
				# array height controls
				if arrayRatio > canvasRatio:
					shrinkRatio = float(self.winfo_height()) / numpyArray.shape[0]
				# array width controls
				else:
					shrinkRatio = float(self.winfo_width()) / numpyArray.shape[1]
								
				numpyArray = cv2.resize(numpyArray, (0,0), fx=shrinkRatio, fy=shrinkRatio)
				
			# Turn the numpy array into an image
			self.image = Image.fromarray(numpyArray.astype(np.uint8))

			# display on the canvas
			self.imagePhoto = ImageTk.PhotoImage(self.image)
			self.create_image((numpyArray.shape[1]/2,numpyArray.shape[0]/2),image=self.imagePhoto)
