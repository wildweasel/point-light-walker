import time
import cv2
import numpy as np
import sys
from tkinter import *
from tkinter import filedialog
import PointLightWalker
import OpenCVCanvas
import threading
import ButtonState

class PointLightWalkerGUI(Tk):
	
	def __init__(self):
		# Call super constructor
		Tk.__init__(self)
		
		# put the window at the top left of the screen
		self.geometry("+0+0")

		# Get the current screen dimensions...
		screen_width = self.winfo_screenwidth()
		screen_height = self.winfo_screenheight()
		# ... and size the display windows appropriately
		windowWidth = screen_width / 3
		windowHeight = (screen_height-200)/ 2
						
		# Build the menu bars
		menu1 = Frame(self)
		menu1.pack()
		menu2 = Frame(self)
		menu2.pack()
		menu3 = Frame(self)
		menu3.pack()
		
		#  Add a load button to the menu bar
		loadButton = Button(menu1, text='Load Video Clip', command=self.loadVideo)
		loadButton.pack(side=LEFT)
		
		#  Add a run button to the menu bar
		runButton = Button(menu1, text='Run', command=self.run)
		runButton.pack(side=LEFT)
		
		#  Allow the user to pause the playback
		pauseButton = Button(menu1, text='Pause', command=self.pause)
		pauseButton.pack(side=LEFT)
		
		#  Register the initial state of the GUI 
		self.buttonState = ButtonState.ButtonState(loadButton, runButton, pauseButton, ButtonState.ButtonState.State.INIT)
				
		# Allow the user some control over the playback speed
		self.delay = StringVar()
		self.delay.set(0)
		Label(menu1, text = "Speed").pack(side=LEFT)
		Spinbox(menu1, from_=0, to=1, increment=.1, textvariable=self.delay).pack(side=LEFT)
		
		# Initial Gaussian blur sigma level.  Kernel size auto-calculated
		self.blurSigmaC = StringVar()
		self.blurSigmaC.set(7)
		Label(menu2, text = "Centroid Blur Sigma").pack(side=LEFT)
		Spinbox(menu2, from_=0, to=15, increment=1, textvariable=self.blurSigmaC).pack(side=LEFT)

		# Size of erode element for motion detect result.  Uses sqaure-shaped element
		self.erodeElementSizeC = StringVar()
		self.erodeElementSizeC.set(3)
		Label(menu2, text = "Centroid Erode Size").pack(side=LEFT)
		Spinbox(menu2, from_=0, to=20, increment=1, textvariable=self.erodeElementSizeC).pack(side=LEFT)

		# Size of dilate element for motion detect result.  Uses sqaure-shaped element
		self.dilateElementSizeC = StringVar()
		self.dilateElementSizeC.set(10)
		Label(menu2, text = "Centroid Dilate Size").pack(side=LEFT)
		Spinbox(menu2, from_=0, to=15, increment=1, textvariable=self.dilateElementSizeC).pack(side=LEFT)

		# minimum value to survive post-dilation
		self.thresholdValueC = StringVar()
		self.thresholdValueC.set(160)
		Label(menu2, text = "Centroid Threshold").pack(side=LEFT)
		Spinbox(menu2, from_=0, to=200, increment=5, textvariable=self.thresholdValueC).pack(side=LEFT)
				
		self.hueBins = StringVar()
		self.hueBins.set(10)
		Label(menu3, text = "Hue Bins").pack(side=LEFT)
		Spinbox(menu3, from_=1, to=20, increment=1, textvariable=self.hueBins).pack(side=LEFT)

		self.hueSlices = StringVar()
		self.hueSlices.set(3)
		Label(menu3, text = "Keep Hue Bins").pack(side=LEFT)
		Spinbox(menu3, from_=1, to=5, increment=1, textvariable=self.hueSlices).pack(side=LEFT)
		
		self.waterShedDistance = StringVar()
		self.waterShedDistance.set(20)
		Label(menu3, text = "Watershed Distance").pack(side=LEFT)
		Spinbox(menu3, from_=2, to=50, increment=2, textvariable=self.waterShedDistance).pack(side=LEFT)
		
		self.dumpOneHue = IntVar()
		Checkbutton(menu3, text="Dump first hue bin", variable=self.dumpOneHue).pack(side=LEFT)
		
		# Display video(s) row
		videoRow1 = Frame(self)
		videoRow1.pack()
		videoRow2 = Frame(self)
		videoRow2.pack()
		
		# Video screens
		self.videoCanvas1 = OpenCVCanvas.OpenCVCanvas(videoRow1, height=windowHeight, width=windowWidth)
		self.videoCanvas1.pack(side=LEFT)
		
		self.videoCanvas2 = OpenCVCanvas.OpenCVCanvas(videoRow1, height=windowHeight, width=windowWidth)
		self.videoCanvas2.pack(side=LEFT)
		
		self.videoCanvas3 = OpenCVCanvas.OpenCVCanvas(videoRow1, height=windowHeight, width=windowWidth)
		self.videoCanvas3.pack(side=LEFT)
		
		self.videoCanvas4 = OpenCVCanvas.OpenCVCanvas(videoRow2, height=windowHeight, width=windowWidth)
		self.videoCanvas4.pack(side=LEFT)
		
		self.videoCanvas5 = OpenCVCanvas.OpenCVCanvas(videoRow2, height=windowHeight, width=windowWidth)
		self.videoCanvas5.pack(side=LEFT)		
		
		self.videoCanvas6 = OpenCVCanvas.OpenCVCanvas(videoRow2, height=windowHeight, width=windowWidth)
		self.videoCanvas6.pack(side=LEFT)
		
		# Instantiate the CV processing object:
		self.pointLightWalker = PointLightWalker.PointLightWalker()
		
		# Initial state of processing thread is empty
		self.t = None
		
	def loadVideo(self):
		# Use the deafault tkinter file open dialog
		self.videoFile = filedialog.askopenfilename()
		
		# make sure the user didn't hit 'Cancel'		
		if self.videoFile:
			#  Start the processing thread
			self.run()
			
	def run(self):
		
		self.buttonState.setState(ButtonState.ButtonState.State.RUNNING)
		
		# If the worker thread is already active (because we came from PAUSED), 
		# 	the change to RUNNING state is all that needs done
		if self.t is not None and self.t.isAlive():		
			return
		# If the worker thread is not already active, it's because we came from
		#	INIT or STOPPED, so we should start it up
						
		# Clear the MOG
		self.pointLightWalker.clear()
		
		# Instantiate an openCV Video capture object
		cap = cv2.VideoCapture(self.videoFile)
			
		# Run the video processing in a background thread
		self.t = threading.Thread(target=self.processVideo, args=(cap,))
		# because this is a daemon, it will die when the main window dies
		self.t.setDaemon(True)
		self.t.start()
				
		
	def pause(self):
		self.buttonState.setState(ButtonState.ButtonState.State.PAUSED)
			
	def processVideo(self, cap):
			
		# This loop works better if it's pre-started
		ret,img = cap.read()
		self.processFrame(img)
				
		while cap.isOpened() and img is not None:

			# If we're paused, just chill
			if self.buttonState.getState() == ButtonState.ButtonState.State.PAUSED:
				continue
											
			# Get the frame and process it
			ret,img = cap.read()
			self.processFrame(img)
			
			# Have we enabled speed control?
			delay = float(self.delay.get())
			if  delay > 0:
				time.sleep(delay)
				
		# Processing is over.
		self.buttonState.setState(ButtonState.ButtonState.State.STOPPED)

	def processFrame(self, img):
		
		# make sure the image is OK
		if img is None:
			return
		
		# OpenCV reads in BGR - switch to RGB
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		# process the frame
		result, step4, step3, step2, step1 = self.pointLightWalker.process(img, int(self.blurSigmaC.get()), int(self.erodeElementSizeC.get()), 
																		   int(self.dilateElementSizeC.get()), int(self.thresholdValueC.get()),
																		   int(self.hueBins.get()), int(self.hueSlices.get()),
																		   int(self.waterShedDistance.get()), self.winfo_screenwidth(), self.winfo_screenheight(),
																		   self.dumpOneHue.get())
		# check to make sure we have something to display
		# (motion detection usually needs to process more than one frame)
		if result is not None:
			self.videoCanvas1.publishArray(img)
			self.videoCanvas2.publishArray(step1)
			self.videoCanvas3.publishArray(step2)
			self.videoCanvas4.publishArray(step3)	
			self.videoCanvas5.publishArray(step4)
			self.videoCanvas6.publishArray(result)	


app = PointLightWalkerGUI()
app.mainloop()
