# Online Inference

This folder contains all of the code required to run Lip Reading in an online scenario on the Jetson TX2. This component captures the online inference portion of the Jetson Lip Reading project for UCB MIDS W251 Section 3 course, authored by Diana Iftimie, Tom Goter, & Noah Pflaum. This work is inspired by the work done for the Lip2Wav project that this project focuses on exploring and expanding.

This lip reading pipeline is a dockerized pipeline that comes in three key parts that are all connected via MQTT:

 * Face Detection (which can additional be mimicked with the Fake Face Detector)
 * Audio Synthesis (the core component of doing the Lip Reading task)
 * Audio Playback (which enables playing synthesized audio on the Jetson)

The three components are outlined in the diagram below:
[PipelineDiagram](./PipelineDiagram.png)

Each remaining section of this README will go into the detailed information for each container.


## Face Detector

### Running the Container

All necessary code for face detection lives in `face_detector/`, including the Dockerfile used to construct this container. 

The following docker commands can be used to create and run everything from the `face_detector/` folder:
```
> docker build -t fd_jlr -f Dockerfile.facedetector .                                      # to build the image
> docker run -ti --name fd1 -e DISPLAY=$DISPLAY -e HOST="10.0.0.47" --privileged fd_jlr    # to start the container
> docker container stop fd1 && docker container rm fd1                                     # to stop & remove container
```

Running the docker container will start up the camera at video input 1 and face detection model and will read in images and crop out the first identified face in each frame. These faces will then be published to the topic specified in the Dockerfile. 

### References & Licenses

Credits for the face detection model go to the MTCNN Face Detection demo linked here: [https://github.com/jkjung-avt/tensorrt_demos#demo-2-mtcnn](https://github.com/jkjung-avt/tensorrt_demos#demo-2-mtcnn)

The referenced demo references source code of [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) samples to develop most of the demos in its repository. Those NVIDIA samples are under [Apache License 2.0](https://github.com/NVIDIA/TensorRT/blob/master/LICENSE).


## Fake Face Detector

To help with the development of the audio synthesizer (so there's nothing dependent on the actual face detector), we had the additional fake face detector for taking data preprocessed from YouTube videos (via the Lip2Wav's original preprocess.py script), which can be used to standardize input data and emulate the actual face detector logic. 

### Running the Container

All necessary code for the fake face detection lives in `fake_face_detector/`, including the Dockerfile used to construct this container.

The following docker commands can be used to create and run everything from the `fake_face_detector/` folder:
```
> docker build -t ffd_jlr -f Dockerfile.fakefacedetector .                                           # to build the image
> docker run -ti --name ffd1 -e HOST="10.0.0.47" -e SOURCE_DIR="./mini_sample_chem/cut-0/" ffd_jlr   # to run with a single data cut
> docker run -ti --name ffd1 -e HOST="10.0.0.47" -e SOURCE_DIR="./mini_sample_chem/" ffd_jlr         # to run with multiple cuts
> docker container stop ffd1 && docker container rm ffd1                                             # to stop & remove container
```

Be sure to have the preprocessed dataset in the `fake_face_detector/` directory, as it will search for the `SOURCE_DIR` relative to this location. The sample mini dataset for the chemistry lecturer used for development can be found on Google Drive [here](https://drive.google.com/drive/folders/1oYmKVGE3cIwWoTnhgZwPhifEY08a9_Kd?usp=sharing).


## Running the Synthesizer

### Running the Container

All necessary code for speech/audio synthesis lives in `audio_synthesizer/`, including the Dockerfile used to construct this container. 

The following docker commands can be used to create and run everything from the `audio_synthesizer/` folder:
```
> docker build -t as_jlr -f Dockerfile.audiosynthesizer .         # to build the image     

# to start the docker container               
> docker run -ti --name as1 -e HOST="10.0.0.47" --privileged \
	-e CHECKPOINT="weights/<path-to-weights>" \
	-e PRESET="synthesizer/presets/<preset-name>.json" as_jlr    

> docker container stop as1 && docker container rm as1            # to stop & remove container
```

Be sure that the you've included the presets you're interested in using in the folder `audio_synthesizer/synthesizer/presets/` and the weights you're interested in using in `audio_synthesizer/weights/` prior to building the container to ensure they will be available at container runtime. Sample weights for the chemistry lecturer can be found on Google Drive [here](https://drive.google.com/drive/folders/17NGz5Tp0wrLGV0Ub6pz_KMfCvo00O4eG?usp=sharing). Presets for the chem lecturer are included in `synthesizer/presets/chem.json` An example of running the docker container with this sample chemistry lecturer can be run as follows:

```
> docker run -ti --name as1 -e HOST="10.0.0.47" -e CHECKPOINT="weights/chem/tacotron_model.ckpt-159000" -e PRESET="synthesizer/presets/chem.json" --privileged as_jlr
```

### References & Licenses

Credits for the work done for synthesizing the audio samples from images of faces goes to the research project Lip2Wav linked here: [https://github.com/Rudrabha/Lip2Wav](https://github.com/Rudrabha/Lip2Wav)

For licenses, citations, and acknowledgements, please refer to the links included below:
* [https://github.com/Rudrabha/Lip2Wav#license-and-citation](https://github.com/Rudrabha/Lip2Wav#license-and-citation)
* [https://github.com/Rudrabha/Lip2Wav#acknowledgements](https://github.com/Rudrabha/Lip2Wav#acknowledgements)


## Runing the Audio Player

### Running the Container

All necessary code for playing the synthesized speech/audio lives in `audio_synthesizer/`, including the Dockerfile used to construct this container. 

```
> docker build -t ap_jlr -f Dockerfile.audioplayer .    # to build the image

# to start the docker container
> docker run -ti --name ap1 -e SUB_HOST="10.0.0.47" \
    --device /dev/snd --privileged \
    -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
    -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
    -v ~/.config/pulse/cookie:/root/.config/pulse/cookie \
    --group-add $(getent group audio | cut -d: -f3) \
    ap_jlr

> docker container stop ap1 && docker container rm ap1   # to stop & remove container
```

Be sure that your Jetson TX2 device either has a speaker connected to it or has sound forwarding via a remote desktop software such as NoMachine to hear the audio playback. If at any point in time you have difficulties getting audio to play, be sure to leverage the `alsa` and `pulseaudio` libraries and try running `aplay /usr/share/sounds/alsa/Front_Center.wav` to help with troubelshooting.