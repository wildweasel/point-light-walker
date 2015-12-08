from enum import Enum

# Simple encapsulation of button state for the PointLightWalkerGUi
class ButtonState():

	class State(Enum):
		INIT = 0
		RUNNING = 1
		PAUSED = 2
		STOPPED = 3
	
	def __init__(self, loadButton, runButton, pauseButton, state):
		self.loadButton = loadButton
		self.runButton = runButton
		self.pauseButton = pauseButton
		self.setState(state)
			
	def setState(self, newState):
		self.state = newState
		if newState == ButtonState.State.INIT:
			self.loadButton.config(state="normal")
			self.runButton.config(state="disabled")
			self.pauseButton.config(state="disabled")
		if newState == ButtonState.State.RUNNING:
			self.loadButton.config(state="disabled")
			self.runButton.config(state="disabled")
			self.pauseButton.config(state="normal")	
		if newState == ButtonState.State.PAUSED:
			self.loadButton.config(state="disabled")
			self.runButton.config(state="normal")
			self.pauseButton.config(state="disabled")	
		if newState == ButtonState.State.STOPPED:
			self.loadButton.config(state="normal")
			self.runButton.config(state="normal")
			self.pauseButton.config(state="disabled")	
		
	def getState(self):
		return self.state
