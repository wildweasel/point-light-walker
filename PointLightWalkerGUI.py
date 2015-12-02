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
		
		# Display video(s) row
		videoRow = Frame(self)
		videoRow.pack()
		
		# Video 
		self.videoCanvas1 = OpenCVCanvas.OpenCVCanvas(videoRow, height=windowHeight, width=windowWidth)
		self.videoCanvas1.pack(side=LEFT)
		
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
				result = self.pointLightWalker.process(img)
				# check to make sure we have something to display
				# (motion detection usually needs to process more than one frame)
				if result is not None:
					self.videoCanvas1.publishArray(result)				
			
			# Have we enabled speed control?
			delay = float(self.delay.get())
			if  delay > 0:
				time.sleep(delay)


app = PointLightWalkerGUI()
app.mainloop()
