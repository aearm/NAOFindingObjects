import sys
import time
import argparse
from naoqi import ALProxy, ALBroker, ALModule
import vision_definitions
import almath
import cv2
import numpy as np
import itertools
import threading
import math

framerate = 15

def pprint(str):
	print("[RobotController] %s" % str)

def pwrite(str):
	sys.stdout.write("[RobotController] %s" % str)
	sys.stdout.flush()

# Class that recieves collision events from the robot
class CollisionDetectorModule(ALModule):
	def __init__(self, name, memory_proxy, callback):
		ALModule.__init__(self, name)

		self.name = name

		self.memoryProxy = memory_proxy
		self.memoryProxy.subscribeToEvent("ALMotion/Safety/MoveFailed",
			name, "onMoveFailure")

		self.callback = callback

		pprint("CollisionDetector initialized.")

	def onMoveFailure(self, eventName, val, subscriberIdentifier):
		sys.stdout.write(" - collision - ")
		sys.stdout.flush()
		self.callback()

	def __del__(self):
		self.memoryProxy.unsubscribeToEvent("ALMotion/Safety/MoveFailed", 
			self.name)
		pprint("CollisionDetector destroyed!")

# Class that simplifies robot startup, shutdown, and control
class RobotController:
	def __init__(self, ip, port, simulation_mode=True):
		self.ip = ip
		self.port = port
		self.camProxy = ALProxy("ALVideoDevice", self.ip, self.port)
		self.postureProxy = ALProxy("ALRobotPosture", self.ip, self.port)
		self.motionProxy = ALProxy("ALMotion", self.ip, self.port)
		self.memoryProxy = ALProxy("ALMemory", self.ip, self.port)
		self.blobProxy = ALProxy("ALColorBlobDetection", self.ip, self.port)
		self.trackerProxy = ALProxy("ALTracker", self.ip, self.port)
		self.ttsProxy = ALProxy("ALTextToSpeech", self.ip, self.port)
		self.broker = ALBroker("pythonMotionBroker", "0.0.0.0", 0, 
			self.ip, self.port)

		if simulation_mode == False:
			self.autolifeProxy = ALProxy("ALAutonomousLife", self.ip, self.port)
			self.autolifeProxy.setState("disabled")

		self.topCamera = self.camProxy.subscribeCamera("topCamera_pycli", 
			0, vision_definitions.kVGA, vision_definitions.kBGRColorSpace,
			framerate)
		self.bottomCamera = self.camProxy.subscribeCamera(
			"bottomCamera_pycli", 1, vision_definitions.kQQVGA, 
			vision_definitions.kBGRColorSpace, framerate)

		self.motionProxy.wakeUp()
		self.enableExternalCollisionProtection()
		self.postureProxy.goToPosture("StandInit", 1.0)

	def stiffnessOn(self):
		# We use the "Body" name to signify the collection of all joints
		pNames = "Body"
		pStiffnessLists = 1.0
		pTimeLists = 1.0
		self.motionProxy.stiffnessInterpolation(pNames, pStiffnessLists, 
			pTimeLists)

	def stiffnessOff(self):
		# We use the "Body" name to signify the collection of all joints
		pNames = "Body"
		pStiffnessLists = 0.0
		pTimeLists = 1.0
		self.motionProxy.stiffnessInterpolation(pNames, pStiffnessLists, 
			pTimeLists)

	def enableArmCollisionProtection(self):
		chainName = "Arms"
		enable  = True
		isSuccess = self.motionProxy.setCollisionProtectionEnabled(chainName, 
			enable)
		pprint("Anticollision activation on arms: " + str(isSuccess))

	def disableArmCollisionProtection(self):
		chainName = "Arms"
		enable  = False
		isSuccess = self.motionProxy.setCollisionProtectionEnabled(chainName, 
			enable)
		pprint("Anticollision deactivation on arms: " + str(isSuccess))

	def enableExternalCollisionProtection(self):
		self.motionProxy.setExternalCollisionProtectionEnabled("All", True)

	def disableExternalCollisionProtection(self):
		self.motionProxy.setExternalCollisionProtectionEnabled("All", False)

	def headRotateAbsolute(self, degrees):
		# Example showing a single target angle for one joint
		# Interpolate the head yaw to 15 degrees in 1.0 second
		names      = "HeadYaw"
		angleLists = degrees*almath.TO_RAD
		timeLists  = 1.0 if (degrees == 0.0) else abs(15.0/degrees)
		isAbsolute = True
		self.motionProxy.angleInterpolation(names, angleLists, timeLists, 
			isAbsolute)

	def headPitchAbsolute(self, degrees):
		# Example showing a single target angle for one joint
		# Interpolate the head pitch to 15 degrees in 1.0 second
		names      = "HeadPitch"
		angleLists = degrees*almath.TO_RAD
		timeLists  = 1.0 if (degrees == 0.0) else abs(15.0/degrees)
		isAbsolute = True
		self.motionProxy.angleInterpolation(names, angleLists, timeLists, 
			isAbsolute)	

	def bodyRotateAbsolute(self, degrees):
		self.motionProxy.moveTo(0.0, 0.0, degrees*almath.TO_RAD)

	def bodyWalkForward(self, distance):
		self.motionProxy.moveTo(distance, 0.0, 0.0)

	def takePicture(self, camera):
		naoImage = self.camProxy.getImageRemote(camera)

		# Get the image size and pixel array.
		imageWidth = naoImage[0]
		imageHeight = naoImage[1]
		timestampUS = naoImage[5]
		array = naoImage[6]
		cameraID = naoImage[7]
		leftAngle = naoImage[8]
		topAngle = naoImage[9]
		rightAngle = naoImage[10]
		bottomAngle = naoImage[11]

		array = np.fromstring(array, dtype=np.uint8)

		# Note: This returns images as BGR
		return {"imageWidth": imageWidth,
			"imageHeight": imageHeight,
			"image": np.reshape(array, (imageHeight, imageWidth, 3)),
			"leftAngle": leftAngle,
			"topAngle": topAngle,
			"rightAngle": rightAngle,
			"bottomAngle": bottomAngle,
			"cameraID": cameraID,
			"timestampUS": timestampUS}

	def releasePicture(self, camera):
		self.camProxy.releaseImage(camera)

	def lookAtPixelInImage(self, image_dict, x, y):
		lookAtX, lookAtY = self.camProxy.getAngularPositionFromImagePosition(
			image_dict["cameraID"],
			[x/image_dict["imageWidth"], y/image_dict["imageHeight"]])
		self.motionProxy.angleInterpolation("HeadYaw", lookAtX, 
			1.0 if (lookAtX == 0.0) else abs((30.0*almath.TO_RAD)/lookAtX), 
			True)
		self.motionProxy.angleInterpolation("HeadPitch", lookAtY, 
			1.0 if (lookAtY == 0.0) else abs((30.0*almath.TO_RAD)/lookAtY), 
			True)

	def rotateBodyToObjectInImage(self, image_dict, x, y):
		lookAtX, _ = self.camProxy.getAngularPositionFromImagePosition(
			image_dict["cameraID"],
			[x/image_dict["imageWidth"], y/image_dict["imageHeight"]])
		self.motionProxy.moveTo(0.0, 0.0, lookAtX)

	def __del__(self):
		self.postureProxy.goToPosture("Sit", 1.0)
		self.disableExternalCollisionProtection()
		self.motionProxy.rest()
		self.camProxy.unsubscribe(self.topCamera)
		self.camProxy.unsubscribe(self.bottomCamera)
		self.broker.shutdown()

class BallDetector(ALModule):
	# RGB values and threshold
	red = (255 - 15, 0, 0, 50)
	blue = (0, 0, 255 - 15, 50)
	yellow = (255 - 15, 255 - 15, 0, 50)

	def __init__(self, name, robot_controller, color="red", diameter=0.2):
		ALModule.__init__(self, name)
		self.name = name
		self.blobDetector = robot_controller.blobProxy
		if color == "red":
			pprint("Looking for a red ball...")
			self.blobDetector.setColor(*self.red)
		elif color == "blue":
			pprint("Looking for a blue ball...")
			self.blobDetector.setColor(*self.blue)
		elif color == "yellow":
			pprint("Looking for a yellow ball...")
			self.blobDetector.setColor(*self.yellow)
		else:
			pwrite(">> Warning << Invalid color set in BallDetector! ")
			print("Defaulting to red...")
			self.blobDetector.setColor(*self.red)
		self.blobDetector.setObjectProperties(20, diameter, "Circle")
		self.memoryProxy = robot_controller.memoryProxy
		self.memoryProxy.subscribeToEvent("ALTracker/ColorBlobDetected",
			self.name, "onBlobDetection")
		self.motionProxy = robot_controller.motionProxy
		self.camProxy = robot_controller.camProxy
		pprint("BallDetector initialized!")
		self.lock = threading.Lock()
		self.info = None

	def onBlobDetection(self, eventName, value, subscriberIdentifier):
		positionInFrameRobot = value[1]
		pprint("Ball detected.")
		timestampUS = value[2][1]
		# The following two actually change over time, so we must get 
		# them every time
		topCameraPositionInFrameRobot = self.motionProxy.getPosition(
			"CameraTop", 2, True)
		bottomCameraPositionInFrameRobot = self.motionProxy.getPosition(
			"CameraBottom", 2, True)
		# CameraTop
		differenceInPos = np.array(positionInFrameRobot) - np.array(
			topCameraPositionInFrameRobot)
		cameraXAngle = differenceInPos[5]
		cameraYAngle = differenceInPos[4]
		topCameraX, topCameraY = self.camProxy.getImagePositionFromAngularPosition(
			0, (cameraXAngle, cameraYAngle))
		# CameraBottom
		differenceInPos = np.array(positionInFrameRobot) - np.array(
			bottomCameraPositionInFrameRobot)
		cameraXAngle = differenceInPos[5]
		cameraYAngle = differenceInPos[4]
		bottomCameraX, bottomCameraY = self.camProxy.getImagePositionFromAngularPosition(
			1, (cameraXAngle, cameraYAngle))
		# Writeback
		self.lock.acquire()
		self.info = {"timestampUS": timestampUS, 
			"positionInFrameRobot": positionInFrameRobot,
			"topCameraNormalizedX": 
				topCameraX if (0.0 <= topCameraX <= 1.0) else None,
			"topCameraNormalizedY": 
				topCameraY if (0.0 <= topCameraY <= 1.0) else None,
			"bottomCameraNormalizedX": 
				bottomCameraX if (0.0 <= bottomCameraX <= 1.0) else None,
			"bottomCameraNormalizedY": 
				bottomCameraY if (0.0 <= bottomCameraY <= 1.0) else None,
			"isInTopCamera":
				(0.0 <= topCameraX <= 1.0) and (
				0.0 <= topCameraY <= 1.0),
			"isInBottomCamera":
				(0.0 <= bottomCameraX <= 1.0) and (
				0.0 <= bottomCameraY <= 1.0)}
		self.lock.release()

	def changeColor(self, color):
		self.blobDetector.setColor(color)

	def getBallInfo(self):
		self.lock.acquire()
		out = self.info
		self.info = None
		self.lock.release()
		return out

	def __del__(self):
		self.memoryProxy.unsubscribeToEvent("ALTracker/ColorBlobDetected", 
			self.name)
		pprint("BallDetector destroyed!")

# The thread that controls the robot's motors and movement
class MotorController(threading.Thread):
	def __init__(self, robot_controller):
		threading.Thread.__init__(self)
		self.daemon = True
		self.robotController = robot_controller
		self.motionProxy = robot_controller.motionProxy
		self.stoppingEvent = threading.Event()
		self.targetLocation = None
		self.hadTargetOnLastIteration = True
		self.numberOfIterationsWithoutTarget = 0
		self.foundObjectCandidate = False
		self.lock = threading.Lock()
		self.motionProxy.moveInit()

	def run(self):
		# Initialize the robot location
		robotlocation = self.motionProxy.getRobotPosition(True)
		# Give some time for the first frame to be processed...
		# The ball might be right in front of you!
		time.sleep(1)
		while True:
			# Lock to make sure only one person accesses the self.targetLocation
			# data at a time 
			self.lock.acquire()
			# Get the new location to calcuate the distance traveled since the
			# last iteration of the loop
			newrobotlocation = self.motionProxy.getRobotPosition(True)
			# Calculate the distance traveled
			distancetraveled = [a - b for a, b in zip(newrobotlocation, 
				robotlocation)]
			# Update the robot's recorded location
			robotlocation = newrobotlocation
			# If the main thread wants this thread to stop, then stop it
			if self.stoppingEvent.isSet():
				try:
					self.lock.release()
				except threading.ThreadError:
					# The lock could already be unlocked. 
					pass
				break
			# If the target location is valid
			if self.targetLocation != None:
				# Subtract the distance traveled from the target location
				self.targetLocation = tuple([a - b for a, b in zip(
					self.targetLocation, distancetraveled)])
				xdist, ydist, rdist = self.targetLocation[0:3]
				# Calculate the euclidian distance
				distance = math.sqrt(float(xdist * xdist + ydist * ydist))
				print("Distance: %f m" % distance)
				if distance > 1.75:
					#print("Going to: (%f m, %f m, %f rad)" % (xdist, ydist, rdist))
					self.robotController.trackerProxy.lookAt(
						(xdist, ydist, 0.0), 2, 1.0, True)
					# Angles are in radians here...
					self.motionProxy.moveToward(
						xdist/distance,
						ydist/distance,
						rdist/math.pi)
				else:
					# If the object is closer than 1.75, then stop and point to the ball
					if self.motionProxy.moveIsActive():
						self.motionProxy.stopMove()
						# Clear the target position, because we have arrived!
						self.targetLocation = None
						# Rotate that last little bit to face the ball.
						self.motionProxy.moveTo(0.0, 0.0, rdist)
						self.robotController.trackerProxy.lookAt(
							(xdist, ydist, 0.0), 2, 1.0, True)
						self.robotController.trackerProxy.pointAt(
							"RArm", (xdist, ydist, 0.0), 2, 1.0)
						# Wait so the user can see the robot pointing
						time.sleep(3)
						# Release the lock and kill the thread
						self.foundObjectCandidate = True
						self.lock.release()
						break
				self.hadTargetOnLastIteration == True
				try:
					self.lock.release()
				except threading.ThreadError:
					# The lock could already be unlocked. 
					pass
			else:
				try:
					self.lock.release()
				except threading.ThreadError:
					# The lock could already be unlocked. 
					pass
				if self.hadTargetOnLastIteration == True:
					self.motionProxy.stopMove()
					self.numberOfIterationsWithoutTarget = 0

				if self.numberOfIterationsWithoutTarget < 4:
					# Rotate 360 degrees in 90 degree intervals, clockwise
					self.robotController.bodyRotateAbsolute(-90.0)
				else:
					# Walk around the room in a spiral
					self.robotController.bodyWalkForward(0.5 * 2)
					self.robotController.bodyRotateAbsolute(-30.0)

				# Scan with the head
				self.robotController.headRotateAbsolute(-45.0)
				self.robotController.headRotateAbsolute(45.0)
				self.robotController.headRotateAbsolute(0.0)

				self.numberOfIterationsWithoutTarget += 1
				self.hadTargetOnLastIteration = False
			
				self.motionProxy.waitUntilMoveIsFinished()
		self.motionProxy.stopMove()
		pprint("MotorController thread exiting!")
		return

	def postTargetLocation(self, loc):
		self.lock.acquire()
		self.targetLocation = loc
		self.lock.release()

	def stop(self):
		pprint("Trying to stop the MotorController...")
		self.stoppingEvent.set()
		pprint("MotorController stop flag set.")
		try:
			self.lock.release()
		except threading.ThreadError:
			# The lock could already be unlocked. 
			pass
		if self.isAlive():
			self.join()

def main(ip, port, color, diameter, simulation_mode):
	pprint("Starting...")

	# Initialize the robot
	controller = RobotController(ip, port, simulation_mode)

	# Function that defines what should happen when a collision
	# is detected
	def collisionBehavior():
		controller.bodyWalkForward(-0.5)
		controller.bodyRotateAbsolute(-90.0)
		return

	# Only enable the collision detector callback, if we are running
	# a simulation. It doesn't work well on the actual robot.
	if simulation_mode:
		global collisiondetector
		collisiondetector = CollisionDetectorModule("collisiondetector", 
			controller.memoryProxy, collisionBehavior)

	# Initialize the motor control thread
	motorControl = MotorController(controller)

	# Initialize the object that gets callbacks when a ball is detected
	global balldetector
	balldetector = BallDetector("balldetector", controller, color, diameter)

	pprint("Initialized.")

	try:
		# Start the motor control thread
		motorControl.start()
		pwrite("Running"); print("");
		while True:
			#sys.stdout.write(".")
			#sys.stdout.flush()
			topImage = controller.takePicture(controller.topCamera)
			frame = topImage["image"]

			blob = balldetector.getBallInfo()
			if blob != None:
				if blob["isInTopCamera"]:
					# Annotate the top camera image
					x = int(round(blob["topCameraNormalizedX"] * 
						topImage["imageWidth"]))
					y = int(round(blob["topCameraNormalizedY"] * 
						topImage["imageHeight"]))
					cv2.circle(frame, (x, y), 10, (0, 255, 0), 2, 8, 0)
				if blob["isInBottomCamera"]: 
					pprint("Object detected in the bottom camera!")
				targetpos = blob["positionInFrameRobot"]
				# Send the identified target to the motor control thread, so
				# it can move to it
				motorControl.postTargetLocation(
					(targetpos[0], targetpos[1], targetpos[5]))

			# Show the frame in the window
			cv2.imshow("Top Camera", frame)
			cv2.waitKey(1)
			controller.releasePicture(controller.topCamera)

			# If the motor control thread exited, then shut down the program
			if motorControl.isAlive() != True:
				pprint("Object found!")
				pprint("Exiting.")
				cv2.destroyAllWindows()
				return

	except KeyboardInterrupt:
		print(" ")
		pprint("Interrupted by user.")
		pprint("Stopping...")

	motorControl.stop()
	pprint("Joining the MotorController thread...")

	cv2.destroyAllWindows()
	pprint("Stopped.")

	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ip", type=str, default="127.0.0.1",
		help="Robot ip address")
	parser.add_argument("--port", type=int, default=9559,
		help="Robot port number")
	parser.add_argument("--color", dest="color",
		type=str, default="red", help="The color of the ball to search for",
		choices=["red", "blue", "yellow",])
	parser.add_argument("--diameter", dest="diameter", type=float, 
		default=0.2, 
		help="The diameter of the ball to search for (in meters)")
	parser.add_argument("--simulation-mode", dest="simulation_mode",
		action="store_true",
		help="Run the program for a robot in simultion mode")

	args = parser.parse_args()
	main(args.ip, args.port, args.color, args.diameter, args.simulation_mode)
