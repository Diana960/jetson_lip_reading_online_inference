from synthesizer.tacotron2 import Tacotron2 # CPU-based
from synthesizer.tacotron_tpg import Tacotron2 as Tacotron_tpg # GPU-Based

from synthesizer.hparams import hparams
from multiprocess.pool import Pool  # You're free to use either one
#from multiprocessing import Pool   # 
from synthesizer import audio
from pathlib import Path
from typing import Union, List
import tensorflow as tf
import numpy as np
import numba.cuda
import librosa
from profilehooks import timecall


class Synthesizer:
    sample_rate = hparams.sample_rate
    hparams = hparams
    
    def __init__(self, verbose=True, low_mem=False, manual_inference=False):
        """
        Creates a synthesizer ready for inference. The actual model isn't loaded in memory until
        needed or until load() is called.
        
        :param checkpoints_dir: path to the directory containing the checkpoint file as well as the
        weight files (.data, .index and .meta files)
        :param verbose: if False, only tensorflow's output will be printed TODO: suppress them too
        :param low_mem: if True, the model will be loaded in a separate process and its resources 
        will be released after each usage. Adds a large overhead, only recommended if your GPU 
        memory is low (<= 2gb)
        """
        self.verbose = verbose
        self._low_mem = low_mem
        
        # Prepare the model
        self._model_tacotron2 = None  # type: Tacotron2
        self._model_tacotron_tpg = None # type: Tacotron_tpg
        
        #checkpoint_state = tf.train.get_checkpoint_state(checkpoints_dir)
        #if checkpoint_state is None:
        #    raise Exception("Could not find any synthesizer weights under %s" % checkpoints_dir)
        #self.checkpoint_fpath = checkpoint_state.model_checkpoint_path
        #if manual_inference:
        #    self.checkpoint_fpath = self.checkpoint_fpath.replace('/ssd_scratch/cvit/rudra/SV2TTS/', '')
        #    self.checkpoint_fpath = self.checkpoint_fpath.replace('logs-', '')

        #if verbose:
        #    model_name = checkpoints_dir.parent.name.replace("logs-", "")
        #    step = int(self.checkpoint_fpath[self.checkpoint_fpath.rfind('-') + 1:])
        #    print("Found synthesizer \"%s\" trained to step %d" % (model_name, step))
     
    def is_loaded(self, cpu_based=True):
        """
        Whether the model is loaded in GPU memory.
        """
        if (cpu_based):
            return self._model_tacotron2 is not None
        else:
            return self._model_tacotron_tpg is not None
    
    def load(self, cpu_based=True):
        """
        Effectively loads the model to GPU memory given the weights file that was passed in the
        constructor.
        """
        if self._low_mem:
            raise Exception("Cannot load the synthesizer permanently in low mem mode")
        tf.reset_default_graph()
        if (cpu_based):
            self._model_tacotron2 = Tacotron2(None, hparams)
        else:
            self._model_tacotron_tpg = Tacotron_tpg(None, hparams)
            
    def synthesize_spectrograms(self, faces, return_alignments=False):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.
        :param texts: a list of N text prompts to be synthesized
        :param embeddings: a numpy array or list of speaker embeddings of shape (N, 256) 
        :param return_alignments: if True, a matrix representing the alignments between the 
        characters
        and each decoder output step will be returned for each spectrogram
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the 
        sequence length of spectrogram i, and possibly the alignments.
        """
        if not self.is_loaded(cpu_based=True):
            print("@@@@@@@@@@\nLOADING MODEL -- CPU BASED ....\n@@@@@@@@@@")
            self.load(cpu_based=True)
        else:
            print("$$$$$$$$$$\nMODEL ALREADY LOADED -- CPU BASED \n$$$$$$$$$$$")
            
        specs, alignments = self._model_tacotron2.my_synthesize(faces)
        
        return (specs, alignments) if return_alignments else specs

    @timecall(immediate=True)
    def synthesize_wavs(self, faces, return_alignments=False):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.
        :param texts: a list of N text prompts to be synthesized
        :param embeddings: a numpy array or list of speaker embeddings of shape (N, 256) 
        :param return_alignments: if True, a matrix representing the alignments between the 
        characters
        and each decoder output step will be returned for each spectrogram
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the 
        sequence length of spectrogram i, and possibly the alignments.
        """
        if not self.is_loaded(cpu_based=False):
            print("@@@@@@@@@@\nLOADING MODEL -- GPU BASED ....\n@@@@@@@@@@")
            self.load(cpu_based=False)
        else:
            print("$$$$$$$$$$\nMODEL ALREADY LOADED -- GPU BASED \n$$$$$$$$$$$")
        
        wav = self._model_tacotron_tpg.my_synthesize(faces)
        return wav

    @staticmethod
    def _one_shot_synthesize_spectrograms(checkpoint_fpath, texts, embeddings):
        # Load the model and forward the inputs
        tf.reset_default_graph()
        model = Tacotron2(checkpoint_fpath, hparams)
        specs, alignments = model.my_synthesize(embeddings, texts)
        
        # Detach the outputs (not doing so will cause the process to hang)
        specs, alignments = [spec.copy() for spec in specs], alignments.copy()
        
        # Close cuda for this process
        model.session.close()
        numba.cuda.select_device(0)
        numba.cuda.close()
        
        return specs, alignments

    @staticmethod
    def load_preprocess_wav(fpath):
        """
        Loads and preprocesses an audio file under the same conditions the audio files were used to
        train the synthesizer. 
        """
        wav = librosa.load(fpath, hparams.sample_rate)[0]
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        return wav

    @staticmethod
    def make_spectrogram(fpath_or_wav: Union[str, Path, np.ndarray]):
        """
        Creates a mel spectrogram from an audio file in the same manner as the mel spectrograms that 
        were fed to the synthesizer when training.
        """
        if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
            wav = Synthesizer.load_preprocess_wav(fpath_or_wav)
        else:
            wav = fpath_or_wav
        
        mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
        return mel_spectrogram

    @staticmethod
    def griffin_lim(mel):
        """
        Inverts a mel spectrogram using Griffin-Lim. The mel spectrogram is expected to have been built
        with the same parameters present in hparams.py.
        """
        return audio.inv_mel_spectrogram(mel, hparams)
    
