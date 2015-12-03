import time
import cv2
import numpy as np
import sys
from tkinter import *
from tkinter import filedialog
import PointLightWalker
import OpenCVCanvas
import threading

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
						
		# Build a menu bar
		menu = Frame(self)
		menu.pack()
	
		#  Add a load button to the menu bar
		Button(menu, text='Load Video Clip', command=self.loadVideo).pack(side=LEFT)
		
		#  Add a run button to the menu bar
		self.runButton = Button(menu, text='Run', state=DISABLED, command=self.run)
		self.runButton.pack(side=LEFT)
				
		# Allow the user some control over the playback speed
		self.delay = StringVar()
		self.delay.set(0)
		Label(menu, text = "Speed").pack(side=LEFT)
		Spinbox(menu, from_=0, to=1, increment=.1, textvariable=self.delay).pack(side=LEFT)
		
		self.blurSigma = StringVar()
		self.blurSigma.set(3)
		Label(menu, text = "Blur Sigma").pack(side=LEFT)
		Spinbox(menu, from_=0, to=15, increment=1, textvariable=self.blurSigma).pack(side=LEFT)

		self.erodeElementSize = StringVar()
		self.erodeElementSize.set(2)
		Label(menu, text = "Erode Size").pack(side=LEFT)
		Spinbox(menu, from_=0, to=20, increment=1, textvariable=self.erodeElementSize).pack(side=LEFT)

		self.dilateElementSize = StringVar()
		self.dilateElementSize.set(10)
		Label(menu, text = "Dilate Size").pack(side=LEFT)
		Spinbox(menu, from_=0, to=15, increment=1, textvariable=self.dilateElementSize).pack(side=LEFT)

		self.thresholdValue = StringVar()
		self.thresholdValue.set(20)
		Label(menu, text = "Threshold").pack(side=LEFT)
		Spinbox(menu, from_=0, to=40, increment=2, textvariable=self.thresholdValue).pack(side=LEFT)
		
		# Display video(s) row
		videoRow1 = Frame(self)
		videoRow1.pack()
		videoRow2 = Frame(self)
		videoRow2.pack()
		
		# Video 
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

		
	def loadVideo(self):
		# Use the deafault tkinter file open dialog
		self.videoFile = filedialog.askopenfilename()
		
		# make sure the user didn't hit 'Cancel'		
		if self.videoFile:
			# enable the 'Run' button for replays
			self.runButton.config(state="normal")
			#  Start the processing thread
			self.run()
			
	def run(self):
		
		self.pointLightWalker.clear()
		
		# Instantiate an openCV Video capture object
		cap = cv2.VideoCapture(self.videoFile)
			
		# Run the video processing in a background thread
		t = threading.Thread(target=self.processFrame, args=(cap,))
		# because this is a daemon, it will die when the main window dies
		t.setDaemon(True)
		t.start()
			
			
	def processFrame(self, cap):
		
		# Get the frame
		ret,img = cap.read()
				
		while cap.isOpened() and img is not None:
						
			# Get the frame
			ret,img = cap.read()
			# make sure the frame opened OK
			if img is not None:
				
				# OpenCV reads in BGR - switch to RGB
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				
				# process the frame
				result, step4, step3, step2, step1 = self.pointLightWalker.process(img, int(self.blurSigma.get()), int(self.erodeElementSize.get()), int(self.dilateElementSize.get()), int(self.thresholdValue.get()))
				# check to make sure we have something to display
				# (motion detection usually needs to process more than one frame)
				if result is not None:
					self.videoCanvas1.publishArray(img)
					self.videoCanvas2.publishArray(step1)
					self.videoCanvas3.publishArray(step2)
					self.videoCanvas4.publishArray(step3)	
					self.videoCanvas5.publishArray(step4)
					self.videoCanvas6.publishArray(result)			
			
			# Have we enabled speed control?
			delay = float(self.delay.get())
			if  delay > 0:
				time.sleep(delay)


app = PointLightWalkerGUI()
app.mainloop()
