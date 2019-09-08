import pyautogui
import tkinter
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import makepaths
import random
from scipy.signal import savgol_filter

VERSION = 1.0

# --- PARAMETERS ---
refreshrate  = 1	  # refresh rate for screen, positions, and paths (ms)
trackwindow  = 50	  # number of previous mouse positions to keep track of before turning aimbot on
smoothwindow = 41	  # smoothing window for mouse position derivatives (must be odd and less than trackwindow) 
testtime     = 5000   # maximum length of aimtest (ms)
aimbotspeed  = 30	  # speed of aimbot movement (pixels/ms)   [10 - 30] is a good range
splineres    = 200    # number of points to evaluate spline length for

# ------------------

global root 		# tkinter root
global c 			# tkinter canvas
global scrWidth 
global scrHeight


class FakeMouse(object):
	# Initialize parameters, find gradient scaling factors, and initialize spline
	def __init__(self, x, y, vx, vy, ax, ay, xtarget, ytarget):
		self.x = x
		self.y = y
		self.vx = vx
		self.vy = vy
		self.ax = ax
		self.ay = ay
		self.xtarget = xtarget
		self.ytarget = ytarget
		self.spline_xcoeffs = []
		self.spline_ycoeffs = []
		self.length = 0
		self.lengthlist = []
		self.gradscale = self.getGradScaling()
		self.initSpline(self.gradscale)

	# Translating from velocity/acceleration to parametric spline derivatives allows specifying a scaling factor dtau/dt 
	# Generates the scaling based on mouse "momentum": faster mouse velocity/acceleration causes the spline to follow the original path for longer
	# Hardcoded parameters minv,maxv, widthv, etc. give the minimum, maximum scaling factors and a parameter for smoothing between them based on mouse momentum
	# Min dictates continuity of the spline to mouse movements, max dictates overshoot, width dictates the range of speeds between max and min
	def getGradScaling(self):
		eps =  1E-5
		vmag = np.sqrt(self.vx**2 + self.vy**2)
		amag = np.sqrt(self.ax**2 + self.ay**2)

		normv = 1.0/(vmag + eps)
		norma = 1.0/(amag + eps)

		minv = 100
		maxv = 500
		widthv = 5E3
		
		mina = 1
		maxa = 15
		widtha = 5E4

		vscale = minv + (maxv - minv)*np.exp(-widthv/(vmag+eps))
		ascale = mina + (maxa - mina)*np.exp(-widtha/(amag+eps))

		return([vscale*normv, ascale*norma])

	def initSpline(self, gradscale):
		[self.spline_xcoeffs, self.spline_ycoeffs] = makepaths.getSplineCoeffs(self.x, self.y, self.xtarget, self.ytarget, 
														self.vx, self.vy, self.ax, self.ay, gradscale)
		self.updateSplineLength()

	def updateSplineLength(self):
		tau = np.linspace(0,1,splineres)
		oldx = self.x
		oldy = self.y

		s = 0
		slist = []

		for i in range(0,len(tau)):
			[newx, newy] = self.evalSplineAtTau(tau[i])
			ds = np.sqrt((newx - oldx)**2 + (newy - oldy)**2)
			s += ds
			slist.append(s)
			oldx = newx
			oldy = newy

		self.length = s
		self.lengthlist = slist

	def updateTarget(self, xtarget, ytarget):
		self.xtarget = xtarget
		self.ytarget = ytarget
		self.initSpline([1, 1])

	def evalSplineAtTau(self, tau):
		xsum = 0
		ysum = 0
		for i in range(0,len(self.spline_xcoeffs)):
			xsum += self.spline_xcoeffs[i]*tau**i
			ysum += self.spline_ycoeffs[i]*tau**i
		return([xsum, ysum])

	def getSplineDerivativesAtTau(self, tau):
		vx = 0
		vy = 0
		ax = 0
		ay = 0

		for i in range(1,len(self.spline_xcoeffs)):
			vx += i*self.spline_xcoeffs[i]*tau**(i-1)
			vy += i*self.spline_ycoeffs[i]*tau**(i-1)

		for i in range(2,len(self.spline_xcoeffs)):
			ax += (i*(i-1))*self.spline_xcoeffs[i]*tau**(i-2)
			ay += (i*(i-1))*self.spline_ycoeffs[i]*tau**(i-2)

		return([vx,vy,ax,ay])

	def stepAlongSpline(self, tau):
		[newx, newy] = self.evalSplineAtTau(tau)
		[vx,vy,ax,ay] = self.getSplineDerivativesAtTau(tau)
		self.x = newx
		self.y = newy
		self.vx = vx
		self.vy = vy
		self.ax = ax
		self.ay = ay

	def findTauStep(self, ds):
		if (self.lengthlist[-1] < ds):
			return 1.0

		else:
			for i in range(0,len(self.lengthlist)-1):
				if (self.lengthlist[i+1] >= ds and self.lengthlist[i] <= ds):
					dsi = self.lengthlist[i]
					dsf = self.lengthlist[i+1]
					taui = i/(len(self.lengthlist) - 1)
					tauf = (i+1)/(len(self.lengthlist) - 1)

					tau = taui + (tauf - taui)*(ds - dsi)/(dsf - dsi)
					return tau


# -----------------


def point(x,y, color, radius):
	c.create_oval(x-radius, y-radius, x+radius, y+radius, fill=color)

def drawPointAtMousePosition(color, radius):
	x = root.winfo_pointerx() - root.winfo_rootx() # relative position to GUI corner
	y = root.winfo_pointery() - root.winfo_rooty()
	point(x,y,color,radius)

def drawTarget(xcenter, ycenter, width, height, color):
	return c.create_rectangle(int(xcenter - width/2.0), int(ycenter - height/2.0), int(xcenter + width/2.0), int(ycenter + height/2.0), outline = color, fill = color)

def trackMouse(mousedata, ntrack):	
	x = root.winfo_pointerx() - root.winfo_rootx()
	y = root.winfo_pointery() - root.winfo_rooty()
	t = time.time()

	if (len(mousedata) < ntrack):
		mousedata.append([x,y,t])

	else:
		del mousedata[0]
		mousedata.append([x,y,t])

	return mousedata

def getDerivatives(mousedata):
	x, y, t, vx, vy, ax, ay = [], [], [], [], [], [], []

	for i in range(0,len(mousedata)):
		x.append(mousedata[i][0])
		y.append(mousedata[i][1])
		t.append(mousedata[i][2])

	for i in range(0,len(x)-1):
		dx = x[i+1] - x[i]
		dy = y[i+1] - y[i]
		dt = t[i+1] - t[i]
		vx.append(dx/dt)
		vy.append(dy/dt)

	for i in range(0,len(vx)-1):
		dvx = vx[i+1] - vx[i]
		dvy = vy[i+1] - vy[i]
		dt = t[i+2] - t[i+1]
		ax.append(dvx/dt)
		ay.append(dvy/dt)

	vxsmooth = savgol_filter(vx, smoothwindow, 3)
	vysmooth = savgol_filter(vy, smoothwindow, 3)
	axsmooth = savgol_filter(ax, smoothwindow, 3)
	aysmooth = savgol_filter(ay, smoothwindow, 3)

	return([x[-1], y[-1], vx[-1], vy[-1], ax[-1], ay[-1]])


def runAimTest(xtarget_i, ytarget_i, targetwidth, targetheight, targetcolor, targetfreq, targetamp, pointcolor, pointradius, aimbotcutin, ntrack):
	starttime = time.time()
	n = int(testtime/refreshrate)

	aimbot = False
	done = False


	errx_i = (root.winfo_pointerx() - root.winfo_rootx()) - xtarget_i
	erry_i = (root.winfo_pointery() - root.winfo_rooty()) - ytarget_i
	err_i = np.sqrt(errx_i**2 + erry_i**2)

	mousedata = []
	xtarget = xtarget_i
	ytarget = ytarget_i

	for i in range(0,n):
		while (not done):

			target = drawTarget(xtarget, ytarget, targetwidth, targetheight, targetcolor)


			if (aimbot == False):
				drawPointAtMousePosition(pointcolor, pointradius)
				mousedata = trackMouse(mousedata, ntrack)

			else:
				tau = fakemouse.findTauStep(aimbotspeed)

				# Plot 3 intermediary points on the spline between starting point and new point (NOT LINEAR INTERPOLATION)
				for i in range(1,4):
					[x,y] = fakemouse.evalSplineAtTau(i/3.0*tau)
					point(x, y, 'red', 2)

				fakemouse.stepAlongSpline(tau)
				fakemouse.updateTarget(xtarget, ytarget)

				point(fakemouse.x, fakemouse.y, 'red', 2)

				errx = np.abs(fakemouse.x - xtarget)
				erry = np.abs(fakemouse.y - ytarget)
				if (errx < targetwidth/2.0 and erry < targetheight/2.0):
					done = True

			root.update()
			t = time.time() - starttime

			for j in range(0,len(targetfreq[0])):
				fx =   targetfreq[0][j]
				ampx =  targetamp[0][j]				
				fy =   targetfreq[1][j]
				ampy =  targetamp[1][j]			
				xtarget = xtarget_i + ampx*math.sin(t*fx)	
				ytarget = ytarget_i + ampy*math.sin(t*fy)

			errx = (root.winfo_pointerx() - root.winfo_rootx()) - xtarget
			erry = (root.winfo_pointery() - root.winfo_rooty()) - ytarget
			err = np.sqrt(errx**2 + erry**2)

			if (err/err_i < aimbotcutin and not aimbot):
				aimbot = True
				[xstart, ystart, vx, vy, ax, ay] = getDerivatives(mousedata)
				fakemouse = FakeMouse(xstart, ystart, vx, vy, ax, ay, xtarget, ytarget)

			root.after(refreshrate, c.delete(target))


	target = drawTarget(xtarget, ytarget, targetwidth, targetheight, targetcolor)
	root.update()



# ---- INITIALIZE GUI -----
[scrWidth, scrHeight] = pyautogui.size() 

root = tkinter.Tk()
root.resizable(0,0)

c = tkinter.Canvas(root, bg="white", width=scrWidth, height= scrHeight)

c.configure(cursor="crosshair")
c.pack()

freq = [[2], [1]]
amp = [[200], [320]]

xt = random.random()*200 + scrWidth/2.0
yt = random.random()*200 + scrHeight/2.0

runAimTest(xt, yt, 20, 20, 'red', freq, amp, 'blue', 2, 0.5, ntrack = trackwindow)

root.mainloop()

