import pyautogui
import tkinter
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle



VERSION = 1.0
# --- PARAMETERS ---

samplerate = 5 		# sample rate for mouse position (ms)
smoothwindow = 51	# smoothing window for position derivatives  
ntests = 10		# number of mouse movement tests to run

# ------------------

global root 		# tkinter root
global c 			# tkinter canvas
global scrWidth 
global scrHeight
global tempvar		# holds mouse position data

tempvar = []
track = None


def point(x,y, color, radius):
	c.create_oval(x-radius, y-radius, x+radius, y+radius, fill=color)

def drawPointAtMousePosition(color, radius):
	x = root.winfo_pointerx() - root.winfo_rootx() # relative position to GUI corner
	y = root.winfo_pointery() - root.winfo_rooty()
	point(x,y,color,radius)

def plotPath(xlist, ylist, color, radius):
	for i in range(0,len(xlist)):
		point(xlist[i], ylist[i], color, radius)

def plotIdealSpline(xlist, ylist, color):
	interpy = interp1d(xlist, ylist, kind='quadratic')
	xnew = np.concatenate([np.linspace(xlist[1], xlist[2], 20), np.linspace(xlist[0], xlist[1], 20)])
	ynew = interpy(xnew)
	for i in range(0,len(xnew)):
		point(xnew[i], ynew[i], color, 5)



# Creates N points on canvas with given radius and colors. Used for making mouse targets for user
def createNPointSys(n, radius, colorlist):
	xlist = []
	ylist = []

	border = 0.35 # ( 0 - 0.5) generates random points within a border region. 0.2 ignores 20% of the GUI size around the edges (total 40%)

	for i in range(0,n):
		color = colorlist[i]
		x = int((random.random()*(1.0-2.0*border))*scrWidth + border*scrWidth)
		y = int((random.random()*(1.0-2.0*border))*scrHeight + border*scrHeight)
		point(x,y,color, radius)
		xlist.append(x)
		ylist.append(y)

	c.update()
	return([xlist,ylist])

# Toggles global variables for tracking. Bound to left mouse button. Sets time when toggled.
def toggleTracking(event):
	global track 		# boolean var: starts at None then toggled to true then false then stays false afterward.
	global eventtime	
	global TKtrack		# tkinter boolean class - required for TKinter callback

	if (track == None):
		track = True
	elif (track == True):
		track = False
		TKtrack.set(0)	# toggle TKinter variable for wait_variable call

	eventtime = time.time()  # set current time

# Tracks mouse position and time
def trackMotion():
	global track
	global tempvar

	# if true then track, if none then wait, if false do nothing

	if (track == True):
		x = root.winfo_pointerx() - root.winfo_rootx()
		y = root.winfo_pointery() - root.winfo_rooty()
		t = time.time() - eventtime
		tempvar.append([x,y,t])				# append to global mouse position list
		c.update()									# update canvas
		root.after(samplerate, trackMotion)			# resample after samplerate 

	elif (track == None):
		c.update()
		root.after(samplerate, trackMotion)	

# Draws an N point system of targets then tracks mouse movements 
def runNPointTest(n, radius, colorlist, timeout):
	global tempvar
	global track
	global TKtrack
	[targetxpoints, targetypoints] = createNPointSys(n, radius, colorlist)
	tempvar = []

	track = None
	TKtrack.set(1)

	root.bind('<Button-1>', toggleTracking)
	trackMotion()
	root.wait_variable(TKtrack)

	xlist = []
	ylist = []
	tlist = []

	for i in range(0,len(tempvar)):
		xlist.append(tempvar[i][0])
		ylist.append(tempvar[i][1])
		tlist.append(tempvar[i][2])

	plotPath(xlist,ylist, 'red', 2)
#	plotMetrics(xlist,ylist,tlist, smoothwindow)
	return([targetxpoints, targetypoints])




# ---- INITIALIZE GUI -----
[scrWidth, scrHeight] = pyautogui.size() 

root = tkinter.Tk()
root.resizable(0,0)

c = tkinter.Canvas(root, bg="white", width=scrWidth, height= scrHeight)

c.configure(cursor="crosshair")
c.pack()
TKtrack = tkinter.IntVar(root)
TKtrack.set(0)

infotxt = 'Generate mouse paths for database using ' + str(ntests) + ' tests. Click on the green circle to begin tracking, passing \
through blue circles (if any), and end with a click on the red circle. Start and end point of the mouse path does \
not have to be exactly on the circles.'
c.create_text(scrWidth//2, scrHeight//2,fill="black",font="Arial 20 bold",    text=infotxt, width = scrHeight//2 )
root.update()
root.after(10000, c.delete('all'))

# ------------------------



sessid = '%05i' % int(random.random()*10000) 		# Create random 5 digit session ID
timestamp = time.time()
outpath = 'MouseMovementData_SessID' + str(sessid)

data = []
data.append(['Version: ' + str(VERSION), 'Timestamp: ' + str(timestamp), 'Number of tests: ' + str(ntests)]) 



for i in range(0, ntests):
	runtxt = 'Test ' + str(i + 1) + ' of ' + str(ntests)
	c.create_text(scrWidth//2, int(0.05*scrHeight),fill="black",font="Arial 20 bold",    text=runtxt)
	[targetxpoints, targetypoints]  = runNPointTest(3, 25, ['green', 'blue' , 'red'], 9999999)
	root.update()
	data.append([[targetxpoints, targetypoints], tempvar])
	root.after(300, c.delete('all'))

with open(outpath, 'wb') as file:
	pickle.dump(data, file)

root.mainloop()
