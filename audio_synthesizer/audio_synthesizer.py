# System-related & MQTT imports
import time
import sys, os, pickle, argparse, subprocess
from tqdm import tqdm
from profilehooks import timecall
import queue
import paho.mqtt.client as mqtt

# Synthesizer imports
import synthesizer
from synthesizer import inference as sif
import numpy as np
import cv2
from shutil import copy
from glob import glob
from scipy.io import wavfile
import io

##############
# Parameters #
##############

parser = argparse.ArgumentParser()

# Synthesizer params
parser.add_argument("--checkpoint", help="Path to trained checkpoint", required=True)
parser.add_argument("--preset", help="Speaker-specific hyper-params", type=str, required=True)
parser.add_argument("--wav_action", help="What to do with the generated wav files", type=str, required=False, choices=["save", "forward"], default="forward")
parser.add_argument("--results_root", help="Speaker folder path, only needed if wav_action=='save'", required=False)
parser.add_argument("--method_of_synthesis", help="The method of synthesis to used (cpu-based or gpu-based) to generate wav files", type=str, required=False, choices=["cpu", "gpu"], default="gpu")

# Subscribing client params
parser.add_argument("--sub_client_name", help="The name of the MQTT subscribing client", type=str, required=False, default="jetson-face-receiver")
parser.add_argument("--sub_mqtt_host", help="The MQTT host for the subscribing client", type=str, required=True)
parser.add_argument("--sub_mqtt_port", help="The MQTT port for the subscribing client", type=int, required=False, default=1883)
parser.add_argument("--sub_qos", help="The MQTT quality of service for the subscribing client", type=int, required=False, default=2)
parser.add_argument("--sub_topic", help="The MQTT topic the subscribing client should subscribe to", type=str, required=False, default="jetson/faces")

# Publishing client params
parser.add_argument("--pub_client_name", help="The name of the MQTT publishing client", type=str, required=False, default="jetson-audio-sender")
parser.add_argument("--pub_mqtt_host", help="The MQTT host for the publishing client", type=str, required=True)
parser.add_argument("--pub_mqtt_port", help="The MQTT port for the publishing client", type=int, required=False, default=1883)
parser.add_argument("--pub_qos", help="The MQTT quality of service for the publishing client", type=int, required=False, default=2)
parser.add_argument("--pub_topic", help="The MQTT topic the publishing client should publish to", type=str, required=False, default="jetson/audio")

args = parser.parse_args()


##############
# Core Logic #
##############

# Do param-based initializations
with open(args.preset) as f:
   sif.hparams.parse_json(f.read()) ## add speaker-specific parameters
sif.hparams.set_hparam('eval_ckpt', args.checkpoint)

if (args.wav_action == "save"):
   WAVS_ROOT = os.path.join(args.results_root, 'wavs/')
   if not os.path.isdir(WAVS_ROOT):
      os.mkdir(WAVS_ROOT)

# Set params for processing
num_frames = sif.hparams.T

# Define a frame queue 
face_queue = queue.Queue()


def on_log(client, userdata, level, buf):
   print(buf)

def on_connect(client, userdata, flags, rc):
   if (rc == 0):
      print("connected OK")
   else:
      print("Bad connection retruned code = ", rc)
      client.loop_stop()

def on_disconnect(client, userdata, rc):
   print("client disconnected ok")

#def on_publish(client, userdata, mid):
#   print("in on_publish callback mid = ", mid)

def on_subscribe(client, userdata, mid, granted_qos):
   print("subscribed")

def on_message(client, userdata, message):
   # Put frame to python queue to be processed when batches of num_frames are available
   face = np.asarray(bytearray(message.payload), dtype="uint8")
   face = cv2.imdecode(face, cv2.IMREAD_COLOR)
   face_queue.put(face, block=True)


# Set up receiver client & callbacks
receiver_client = mqtt.Client(args.sub_client_name)
receiver_client.on_log = on_log
receiver_client.on_connect = on_connect
receiver_client.on_disconnect = on_disconnect
receiver_client.on_message = on_message
receiver_client.on_subscribe = on_subscribe
receiver_client.connect(args.sub_mqtt_host, args.sub_mqtt_port)


# Set up sender client & callbacks
sender_client = mqtt.Client(args.pub_client_name)
#sender_client.on_log = on_log
sender_client.on_connect = on_connect
sender_client.on_disconnect = on_disconnect
#sender_client.on_publish = on_publish
sender_client.connect(args.pub_mqtt_host, args.pub_mqtt_port)


# start clients & subscribe receiver client to topic
receiver_client.loop_start()
sender_client.loop_start()
receiver_client.subscribe(args.sub_topic, args.sub_qos)

class Generator(object):
   def __init__(self, cpu_based):
      super(Generator, self).__init__()
      self.cpu_based = cpu_based

      self.synthesizer = sif.Synthesizer(verbose=False)
      self.synthesizer.load(cpu_based=self.cpu_based)

      # for CPU-based approach
      self.mel_batches_per_wav_file = 1
      self.mel_batch = None
      self.num_mels = 0

   # Run a single round of inference to force model init
   def force_model_init(self):
      # use the same face for simplicity--the inference results doesn't need to be reasonable
      model_init_face = "./forced_model_init_face.jpg"
      fnames = [model_init_face for i in range(0, num_frames)]
      assert len(fnames) == num_frames

      images = [cv2.imread(fname, cv2.IMREAD_COLOR) for fname in fnames]
      self.generate_wav(images)

   def resize_and_nparrize_images(self, images):
      images = [cv2.resize(img, (sif.hparams.img_size, sif.hparams.img_size)) for img in images]
      images = np.asarray(images) / 255.
      return images

   def post_process_wav(self, wav):
      wav *= 32767 / max(0.01, np.max(np.abs(wav)))
      return wav

   def generate_mel_spec(self, images):
      # Synthesize Spectrogram
      mel_spec = self.synthesizer.synthesize_spectrograms(images)[0]         
         
      # Concatenate batches of mel spectrograms (to get longer wav file samples)
      if self.num_mels == 0:
         self.mel_batch = mel_spec
         self.num_mels = 1
      else:
         self.mel_batch = np.concatenate((self.mel_batch, mel_spec[:, sif.hparams.mel_overlap:]), axis=1)
         self.num_mels += 1

   @timecall(immediate=True)
   def generate_wav_cpu_based(self, images):
      '''
      CPU-based method of converting batches of face images to wav files
      '''
      # Generate mel spectrogram first
      images = self.resize_and_nparrize_images(images)
      self.generate_mel_spec(images)

      # Synthesize wav file from spectrogram when ready
      if (self.num_mels != self.mel_batches_per_wav_file):
         print("not generating wav file yet...")
         return None
      else:
         print("Generating wav file of mel spectrograms")
         wav = self.synthesizer.griffin_lim(self.mel_batch)
         wav = self.post_process_wav(wav)

         self.num_mels = 0
         self.mel_batch = None
         return wav

   @timecall(immediate=True)
   def generate_wav_gpu_based(self, images):
      '''
      GPU-based method of converting batches of face images to wav files
      '''
      images = self.resize_and_nparrize_images(images)
      wav = self.synthesizer.synthesize_wavs(images)
      wav = self.post_process_wav(wav)
      return wav

   def generate_wav(self, images):
      if (self.cpu_based):
         return self.generate_wav_cpu_based(images)
      else:
         return self.generate_wav_gpu_based(images)

   def generate_wav_and_save(self, images, root_dir, wav_num):
      '''
      Generates wav files from batches of images and saves the wav to an output file
      '''
      wav = self.generate_wav(images)
      if (wav is None):
         return # not ready yet
      else:
         print("saving wav file")
         outfile = '{}{}.wav'.format(root_dir, wav_num)
         sif.audio.save_wav(wav, outfile, sr=sif.hparams.sample_rate)

   # Inspiration from here: https://gist.github.com/hadware/8882b980907901426266cb07bfbfcd20
   def generate_wav_and_forward(self, images, mqtt_client, topic, qos):
      '''
      Generates wav files from batches of images and forwards them via MQTT
      '''
      wav = self.generate_wav(images)
      if (wav is None):
         return # not ready yet
      else:
         print("forwarding wav file via MQTT")
         byte_io = io.BytesIO(bytes())
         wavfile.write(byte_io, sif.hparams.sample_rate, wav)
         wav_bytes = byte_io.read()
         mqtt_client.publish(topic, payload=wav_bytes, qos=qos)


def process_faces():
   # Initialize audio generator
   generator = Generator(cpu_based = args.method_of_synthesis == "cpu")
   generator.force_model_init()

   # Wait for messages until disconnected by system interrupt
   print("\n########################\n Ready to receive faces \n########################\n")
   audio_sample_num = 1
   num_frames = sif.hparams.T
   while True:
      print("queue size = " + str(face_queue.qsize()))

      # Check to see if queue has enough frames
      if (face_queue.qsize() >= num_frames):
         print("reached " + str(num_frames) + " frames")

         # Fetch num_frames faces to process from the queue
         faces_to_process = []
         while (len(faces_to_process) != num_frames):
            faces_to_process.append(face_queue.get(block=True))

         # Process frames and generate synthesized audio as wav file data
         # Save as wav file or forward via mqtt
         try:
            if (args.wav_action == "save"):
               generator.generate_wav_and_save(faces_to_process, WAVS_ROOT, audio_sample_num)
            elif (args.wav_action == "forward"):
               generator.generate_wav_and_forward(faces_to_process, sender_client, args.pub_topic, args.pub_qos)

            audio_sample_num += 1
         except KeyboardInterrupt:
            exit(0)
         '''
         except Exception as e:
            print(e)
            continue
         '''

# Run the process face function
process_faces()
