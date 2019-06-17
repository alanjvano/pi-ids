from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import argparse
import warnings
import json
import datetime
import imutils
import uuid
import os
import numpy
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.utils import COMMASPACE, formatdate
from os.path import basename
import copy

# construct the argument parser
parser = argparse.ArgumentParser(description="PiCam IDS")
parser.add_argument("-c", "--conf", required=True, help = "path to the JSON configuration file")
args = vars(parser.parse_args())
print(args)

class TempImage:
	def __init__(self, basePath="./", ext=".jpg"):
		# construct the file path

		date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
		self.path = "{base_path}/pi_images/{date}{ext}".format(base_path=basePath, date=date,ext=ext)

	def cleanup(self):

		# remove the file
		os.remove(self.path)

def send_mail(send_from, send_to, subject, text, files, server):
	#assert isinstance(send_to, list)

	msg = MIMEMultipart()
	msg['From'] = send_from
	msg['To'] = COMMASPACE.join(send_to)

	msg['Date'] = formatdate(localtime=True)
	msg['Subject'] = subject


	msg.attach(MIMEText(text))


	for f in files or []:
		with open(f, "rb") as fil:
			part = MIMEApplication(fil.read(),Name=basename(f))
		# After the file is closed
		part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
		msg.attach(part)


	smtp = smtplib.SMTP(server,587)
	smtp.starttls()
	smtp.login(send_from,conf["email_pwd"])
	smtp.sendmail(send_from, send_to, msg.as_string())
	smtp.close()

# filter warning and load configuration file

warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))

# initialize the camera and variables
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size = tuple(conf["resolution"]))
print("camera warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastSent = datetime.datetime.now()
motionCounter = 0

# begin camera capture
# print "beginning detection"
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image = frame.array
        #origImg = image.copy()
#	imageOrig = image.copy()
	timestamp = datetime.datetime.now()
	text = "none detected"

	# correctly format the frame from processing
	image = imutils.rotate(image, angle=270)
        if not conf["hd"]:
	    image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	gray = cv2.GaussianBlur(gray, (21,21), 0)

	# initialize the average frame
	if avg is None:
		print("starting background model...")
		avg = gray.copy().astype("float")
		rawCapture.truncate(0)
		continue


	# accumulate the weighted average between the current and previous
	# frame, then calculte difference between current and running avg
	cv2.accumulateWeighted(gray, avg, 0.5)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        #print("frame delta: " + str(frameDelta))
        #print("size: " + str(len(frameDelta)))

	# threshold the delta image, dilate the threshold to fill in holes,
	# then find contours on thresholded image
	thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < conf["min_area"]:
			continue

		# compute the bounding box for contour and show box+text
		rawImage = image.copy()
		(x,y,w,h) = cv2.boundingRect(c)
		cv2.rectangle(image,(x,y),(x+w,y+h), (255,255,255), 1)
                text = "motion detected"
                print("contour area: " + str(cv2.contourArea(c)))

	# draw the text and timestamp on the frame

	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	#cv2.putText(image, "Status: {}".format(text), (10,20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
	cv2.putText(image, ts, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1)

	if text == "motion detected":
                print "motion detected!"
		# check last upload time
		if (timestamp - lastSent).seconds >= conf["min_upload_seconds"]:
			motionCounter += 1

			# check if consistent motion high enough
			if motionCounter >= conf["min_motion_frames"]:
				# save the image to a temporary file
				t = TempImage()
				print(t)
				cv2.imwrite(t.path, image)
                                #cv2.imwrite("test_orig.jpg", origImg)
				print(t.path)
				date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
				ext = ".jpg"
		                cv2.imwrite("./pi_images_raw/{date}{ext}".format(date=date,ext=ext),rawImage)
				print("raw copy saved")

				if conf["use_email"]:
					# email the image
					try:

						#filename = "test_file.txt"
						#part = MIMEBase('application',"octet-stream")
						#part.set_payload(open(filename,"rb").read())
						#encoders.encode_base64(part)
						#part.add_header('Content-Disposition','attachment; filename="test_file.txt"')
						#msg.attach(part)

						send_mail(conf["email_account"],conf["dest_email_account"],'motion detected','RPI detected motion.',[t.path],'smtp.gmail.com')
						print("message sent")

					except:
						print("error sending the email...")

				# update the upload timestamp and reset counter
				lastSent = timestamp
				motionCounter = 0

	else:
		motionCounter = 0

	# check to display on screen
	if conf["show_video"]:
		cv2.imshow("Video Feed", image)
		if cv2.waitKey(1) & 0xff == ord("q"):
			break

	# clear the stream for the next frame
	rawCapture.truncate(0)
