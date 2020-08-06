# General Imports
import time
import sys
import os
from os import listdir, path
import argparse

# Image Processing / Face Detection Imports
import numpy as np
import cv2
from mtcnn.mtcnn import TrtMtcnn

# MQTT imports
import paho.mqtt.client as mqtt
mqtt.Client.connected_flag = False


##############
# Parameters #
##############

parser = argparse.ArgumentParser()

# Publishing client params
parser.add_argument("--pub_client_name", help="The name of the MQTT publishing client", type=str, required=False, default="jetson-face-sender")
parser.add_argument("--pub_mqtt_host", help="The MQTT host for the publishing client", type=str, required=True)
parser.add_argument("--pub_mqtt_port", help="The MQTT port for the publishing client", type=int, required=False, default=1883)
parser.add_argument("--pub_qos", help="The MQTT quality of service for the publishing client", type=int, required=False, default=2)
parser.add_argument("--pub_topic", help="The MQTT topic the publishing client should publish to", type=str, required=False, default="jetson/faces")

# Params for mtcnn model
parser.add_argument("--mtcnn_min_size", help="The minimum face pixel width/size for the mtcnn model", type=int, required=False, default=100)

args = parser.parse_args()

##############
# Core Logic #
##############

#def on_log(client, userdata, level, buf):
#   print(buf)

def on_connect(client, userdata, flags, rc):
   if (rc == 0):
      client.connected_flag=True # set flag
      print("connected OK")
   else:
      print("Bad connection retruned code = ", rc)
      client.loop_stop()

def on_disconnect(client, userdata, rc):
   print("client disconnected ok")

#def on_publish(client, userdata, mid):
#   print("in on_publish callback mid = ", mid)


# Set up publishing client & callbacks
client = mqtt.Client(args.pub_client_name)
#client.on_log = on_log
client.on_connect = on_connect
client.on_disconnect = on_disconnect
#client.on_publish = on_publish
client.connect(args.pub_mqtt_host, args.pub_mqtt_port)

# wait for client to establish connection to broker
client.loop_start()
while not client.connected_flag:
   print("waiting to connect...")
   time.sleep(1)


# Define camera to use for capturing images to analyze (1 corresponds to USB camera)
cam = cv2.VideoCapture(1)

# start up face detector and publish message to broker when a face is detected
face_detector = TrtMtcnn()
frame_counter = -1
while(True):
   # capture frame 
   ret, frame = cam.read()

   # identity faces
   start = time.time()
   faces, landmarks = face_detector.detect(frame, minsize=args.mtcnn_min_size)
   duration = time.time() - start
   print("duration = " + str(duration * 1000) + " ms (" + str(1 / duration) + " fps)")
   frame_counter += 1

   # extract & publish faces
   for bb in faces:
      x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
      face = frame[y1:y2, x1:x2]

      rc, png = cv2.imencode('.png', face)
      message = png.tobytes()
      client.publish(args.pub_topic, payload=message, qos=args.pub_qos)

      if (len(faces) > 1):
         break; 


# do clean up
cam.release()
client.loop_stop()
client.disconnect()
