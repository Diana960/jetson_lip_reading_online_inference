# General Imports
import time
import sys
import os
from os import listdir, path
import argparse

# Face Grabbing Dependencies
import cv2
import numpy as np

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

# Params for directory of fake face data
parser.add_argument("--source_directory", help="The source directory of fake data; must be either the full directory of cut directories or an individual cut directory", type=str, required=True)

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


# Specify tuples of source directories of face frames and the corresponding numerical order
# Note that this is specific to how Lip2Wav preprocessing works on YouTube videos
if ("cut" in os.path.basename(os.path.normpath(args.source_directory))):
   # path to single cut directory with face frames specified
   cutdirs_and_nums = [(args.source_directory, 0)]
else:
   # path to multiple cut directories each with own set of face frames specified
   cutdirs_and_nums = [(path.join(args.source_directory, d), int(d[4:])) for d in listdir(args.source_directory) if os.path.isdir(path.join(args.source_directory, d))] 
   cutdirs_and_nums.sort(key=lambda x: x[1])

# Iterate through all cut directories in numerical order (pre-sorted)
for (cutdir, dirnum) in cutdirs_and_nums:
   # Grab all image file names in numerical order
   fnames_and_nums = [(path.join(cutdir, f), int(f[0:-4])) for f in listdir(cutdir) if f[-4:] == ".jpg"] 
   fnames_and_nums.sort(key=lambda x: x[1])

   # start up face grabber and publish message to broker when a face is detected
   for (fname, num) in fnames_and_nums:
      start = time.time()
      # extract & publish faces
      face = cv2.imread(fname, cv2.IMREAD_COLOR)
      if np.shape(face) == ():
         print("continuing due to invalid file ==> fname = " + str(fname) + ", " + str(num))
         continue

      rc, png = cv2.imencode('.png', face)
      message = png.tobytes()
      client.publish(args.pub_topic, payload=message, qos=args.pub_qos)
      
      duration = time.time() - start
      print("fname = " + str(fname) + ", " + str(num), " [duration = " + str(duration * 1000) + " ms (" + str(1/duration) + " fps)]")

# do clean up
client.loop_stop()
client.disconnect()
