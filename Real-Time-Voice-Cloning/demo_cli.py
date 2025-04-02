# import argparse
# import os
# from pathlib import Path

# import librosa
# import numpy as np
# import soundfile as sf
# import torch

# from encoder import inference as encoder
# from encoder.params_model import model_embedding_size as speaker_embedding_size
# from synthesizer.inference import Synthesizer
# from utils.argutils import print_args
# from utils.default_models import ensure_default_models
# from vocoder import inference as vocoder

# import numpy as np
# if not hasattr(np, 'float'):
#     np.float = float

# import numpy as np
# if not hasattr(np, 'complex'):
#     np.complex = complex


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument("-e", "--enc_model_fpath", type=Path,
#                         default="saved_models/default/encoder.pt",
#                         help="Path to a saved encoder")
#     parser.add_argument("-s", "--syn_model_fpath", type=Path,
#                         default="saved_models/default/synthesizer.pt",
#                         help="Path to a saved synthesizer")
#     parser.add_argument("-v", "--voc_model_fpath", type=Path,
#                         default="saved_models/default/vocoder.pt",
#                         help="Path to a saved vocoder")
#     parser.add_argument("--cpu", action="store_true", help=\
#         "If True, processing is done on CPU, even when a GPU is available.")
#     parser.add_argument("--no_sound", action="store_true", help=\
#         "If True, audio won't be played.")
#     parser.add_argument("--seed", type=int, default=None, help=\
#         "Optional random number seed value to make toolbox deterministic.")
#     args = parser.parse_args()
#     arg_dict = vars(args)
#     print_args(args, parser)

#     # Hide GPUs from Pytorch to force CPU processing
#     if arg_dict.pop("cpu"):
#         os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#     print("Running a test of your configuration...\n")

#     if torch.cuda.is_available():
#         device_id = torch.cuda.current_device()
#         gpu_properties = torch.cuda.get_device_properties(device_id)
#         ## Print some environment information (for debugging purposes)
#         print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
#             "%.1fGb total memory.\n" %
#             (torch.cuda.device_count(),
#             device_id,
#             gpu_properties.name,
#             gpu_properties.major,
#             gpu_properties.minor,
#             gpu_properties.total_memory / 1e9))
#     else:
#         print("Using CPU for inference.\n")

#     ## Load the models one by one.
#     print("Preparing the encoder, the synthesizer and the vocoder...")
#     ensure_default_models(Path("saved_models"))
#     encoder.load_model(args.enc_model_fpath)
#     synthesizer = Synthesizer(args.syn_model_fpath)
#     vocoder.load_model(args.voc_model_fpath)


#     ## Run a test
#     print("Testing your configuration with small inputs.")
#     # Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
#     # sampling rate, which may differ.
#     # If you're unfamiliar with digital audio, know that it is encoded as an array of floats
#     # (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
#     # The sampling rate is the number of values (samples) recorded per second, it is set to
#     # 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
#     # to an audio of 1 second.
#     print("\tTesting the encoder...")
#     encoder.embed_utterance(np.zeros(encoder.sampling_rate))

#     # Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
#     # returns, but here we're going to make one ourselves just for the sake of showing that it's
#     # possible.
#     embed = np.random.rand(speaker_embedding_size)
#     # Embeddings are L2-normalized (this isn't important here, but if you want to make your own
#     # embeddings it will be).
#     embed /= np.linalg.norm(embed)
#     # The synthesizer can handle multiple inputs with batching. Let's create another embedding to
#     # illustrate that
#     embeds = [embed, np.zeros(speaker_embedding_size)]
#     texts = ["test 1", "test 2"]
#     print("\tTesting the synthesizer... (loading the model will output a lot of text)")
#     mels = synthesizer.synthesize_spectrograms(texts, embeds)

#     # The vocoder synthesizes one waveform at a time, but it's more efficient for long ones. We
#     # can concatenate the mel spectrograms to a single one.
#     mel = np.concatenate(mels, axis=1)
#     # The vocoder can take a callback function to display the generation. More on that later. For
#     # now we'll simply hide it like this:
#     no_action = lambda *args: None
#     print("\tTesting the vocoder...")
#     # For the sake of making this test short, we'll pass a short target length. The target length
#     # is the length of the wav segments that are processed in parallel. E.g. for audio sampled
#     # at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
#     # 0.5 seconds which will all be generated together. The parameters here are absurdly short, and
#     # that has a detrimental effect on the quality of the audio. The default parameters are
#     # recommended in general.
#     vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

#     print("All test passed! You can now synthesize speech.\n\n")


#     ## Interactive speech generation
#     print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
#           "show how you can interface this project easily with your own. See the source code for "
#           "an explanation of what is happening.\n")

#     print("Interactive generation loop")
#     num_generated = 0
#     while True:
#         try:
#             # Get the reference audio filepath
#             message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
#                       "wav, m4a, flac, ...):\n"
#             in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))

#             ## Computing the embedding
#             # First, we load the wav using the function that the speaker encoder provides. This is
#             # important: there is preprocessing that must be applied.

#             # The following two methods are equivalent:
#             # - Directly load from the filepath:
#             preprocessed_wav = encoder.preprocess_wav(in_fpath)
#             # - If the wav is already loaded:
#             original_wav, sampling_rate = librosa.load(str(in_fpath))
#             preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
#             print("Loaded file succesfully")

#             # Then we derive the embedding. There are many functions and parameters that the
#             # speaker encoder interfaces. These are mostly for in-depth research. You will typically
#             # only use this function (with its default parameters):
#             embed = encoder.embed_utterance(preprocessed_wav)
#             print("Created the embedding")


#             ## Generating the spectrogram
#             text = input("Write a sentence (+-20 words) to be synthesized:\n")

#             # If seed is specified, reset torch seed and force synthesizer reload
#             if args.seed is not None:
#                 torch.manual_seed(args.seed)
#                 synthesizer = Synthesizer(args.syn_model_fpath)

#             # The synthesizer works in batch, so you need to put your data in a list or numpy array
#             texts = [text]
#             embeds = [embed]
#             # If you know what the attention layer alignments are, you can retrieve them here by
#             # passing return_alignments=True
#             specs = synthesizer.synthesize_spectrograms(texts, embeds)
#             spec = specs[0]
#             print("Created the mel spectrogram")


#             ## Generating the waveform
#             print("Synthesizing the waveform:")

#             # If seed is specified, reset torch seed and reload vocoder
#             if args.seed is not None:
#                 torch.manual_seed(args.seed)
#                 vocoder.load_model(args.voc_model_fpath)

#             # Synthesizing the waveform is fairly straightforward. Remember that the longer the
#             # spectrogram, the more time-efficient the vocoder.
#             generated_wav = vocoder.infer_waveform(spec)


#             ## Post-generation
#             # There's a bug with sounddevice that makes the audio cut one second earlier, so we
#             # pad it.
#             generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

#             # Trim excess silences to compensate for gaps in spectrograms (issue #53)
#             generated_wav = encoder.preprocess_wav(generated_wav)

#             # Play the audio (non-blocking)
#             if not args.no_sound:
#                 import sounddevice as sd
#                 try:
#                     sd.stop()
#                     sd.play(generated_wav, synthesizer.sample_rate)
#                 except sd.PortAudioError as e:
#                     print("\nCaught exception: %s" % repr(e))
#                     print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
#                 except:
#                     raise

#             # Save it on the disk
#             filename = "demo_output_%02d.wav" % num_generated
#             print(generated_wav.dtype)
#             sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
#             num_generated += 1
#             print("\nSaved output as %s\n\n" % filename)


#         except Exception as e:
#             print("Caught exception: %s" % repr(e))
#             print("Restarting\n")


# import argparse
# import os
# from pathlib import Path

# import librosa
# import numpy as np
# import soundfile as sf
# import torch

# from encoder import inference as encoder
# from encoder.params_model import model_embedding_size as speaker_embedding_size
# from synthesizer.inference import Synthesizer
# from utils.argutils import print_args
# from utils.default_models import ensure_default_models
# from vocoder import inference as vocoder

# import numpy as np
# if not hasattr(np, 'float'):
#     np.float = float

# import numpy as np
# if not hasattr(np, 'complex'):
#     np.complex = complex

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument("-e", "--enc_model_fpath", type=Path,
#                         default="saved_models/default/encoder.pt",
#                         help="Path to a saved encoder")
#     parser.add_argument("-s", "--syn_model_fpath", type=Path,
#                         default="saved_models/default/synthesizer.pt",
#                         help="Path to a saved synthesizer")
#     parser.add_argument("-v", "--voc_model_fpath", type=Path,
#                         default="saved_models/default/vocoder.pt",
#                         help="Path to a saved vocoder")
#     parser.add_argument("--cpu", action="store_true", help="If True, processing is done on CPU, even when a GPU is available.")
#     parser.add_argument("--no_sound", action="store_true", help="If True, audio won't be played.")
#     parser.add_argument("--seed", type=int, default=None, help="Optional random number seed value to make toolbox deterministic.")
#     args = parser.parse_args()
#     arg_dict = vars(args)
#     print_args(args, parser)

#     # Hide GPUs from Pytorch to force CPU processing
#     if arg_dict.pop("cpu"):
#         os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#     print("Running a test of your configuration...\n")

#     if torch.cuda.is_available():
#         device_id = torch.cuda.current_device()
#         gpu_properties = torch.cuda.get_device_properties(device_id)
#         ## Print some environment information (for debugging purposes)
#         print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
#               "%.1fGb total memory.\n" %
#               (torch.cuda.device_count(),
#                device_id,
#                gpu_properties.name,
#                gpu_properties.major,
#                gpu_properties.minor,
#                gpu_properties.total_memory / 1e9))
#     else:
#         print("Using CPU for inference.\n")

#     ## Load the models one by one.
#     print("Preparing the encoder, the synthesizer and the vocoder...")
#     ensure_default_models(Path("saved_models"))
#     encoder.load_model(args.enc_model_fpath)
#     synthesizer = Synthesizer(args.syn_model_fpath)
#     vocoder.load_model(args.voc_model_fpath)

#     ## Run a test
#     print("Testing your configuration with small inputs.")
#     print("\tTesting the encoder...")
#     encoder.embed_utterance(np.zeros(encoder.sampling_rate))

#     embed = np.random.rand(speaker_embedding_size)
#     embed /= np.linalg.norm(embed)
#     embeds = [embed, np.zeros(speaker_embedding_size)]
#     texts = ["test 1", "test 2"]
#     print("\tTesting the synthesizer... (loading the model will output a lot of text)")
#     mels = synthesizer.synthesize_spectrograms(texts, embeds)

#     mel = np.concatenate(mels, axis=1)
#     no_action = lambda *args: None
#     print("\tTesting the vocoder...")
#     vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

#     print("All test passed! You can now synthesize speech.\n\n")

#     print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
#           "show how you can interface this project easily with your own. See the source code for "
#           "an explanation of what is happening.\n")

#     print("Interactive generation loop")
#     num_generated = 0
#     while True:
#         try:
#             message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, wav, m4a, flac, ...):\n"
#             in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))

#             ## Computing the embedding
#             preprocessed_wav = encoder.preprocess_wav(in_fpath)
#             original_wav, sampling_rate = librosa.load(str(in_fpath))
#             preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
#             print("Loaded file succesfully")

#             embed = encoder.embed_utterance(preprocessed_wav)
#             print("Created the embedding")

#             ## Generating the spectrogram
#             text = input("Write a sentence (+-20 words) to be synthesized:\n")
#             if args.seed is not None:
#                 torch.manual_seed(args.seed)
#                 synthesizer = Synthesizer(args.syn_model_fpath)

#             texts = [text]
#             embeds = [embed]
#             specs = synthesizer.synthesize_spectrograms(texts, embeds)
#             spec = specs[0]
#             print("Created the mel spectrogram")

#             ## Generating the waveform
#             print("Synthesizing the waveform:")
#             if args.seed is not None:
#                 torch.manual_seed(args.seed)
#                 vocoder.load_model(args.voc_model_fpath)

#             generated_wav = vocoder.infer_waveform(spec)

#             ## Post-generation
#             # There is a bug with sounddevice that makes the audio cut one second earlier, so we pad it.
#             generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
#             # Trim excess silences to compensate for gaps in spectrograms (issue #53)
#             generated_wav = encoder.preprocess_wav(generated_wav)

#             # --- Noise Reduction Step ---
#             try:
#                 import noisereduce as nr
#                 # Apply noise reduction. Depending on your audio, you might need to adjust parameters.
#                 # Here, we assume the entire generated_wav can be processed as is.
#                 cleaned_wav = nr.reduce_noise(y=generated_wav, sr=synthesizer.sample_rate)
#                 generated_wav = cleaned_wav
#                 print("Applied noise reduction to clean the audio.")
#             except ImportError:
#                 print("The 'noisereduce' package is not installed. Skipping noise reduction. Install it via 'pip install noisereduce'.")

#             # Play the audio (non-blocking)
#             if not args.no_sound:
#                 import sounddevice as sd
#                 try:
#                     sd.stop()
#                     sd.play(generated_wav, synthesizer.sample_rate)
#                 except sd.PortAudioError as e:
#                     print("\nCaught exception: %s" % repr(e))
#                     print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
#                 except:
#                     raise

#             # Save it on the disk
#             filename = "demo_output_%02d.wav" % num_generated
#             print(generated_wav.dtype)
#             sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
#             num_generated += 1
#             print("\nSaved output as %s\n\n" % filename)

#         except Exception as e:
#             print("Caught exception: %s" % repr(e))
#             print("Restarting\n")


# import argparse
# import os
# from pathlib import Path

# import librosa
# import numpy as np
# import soundfile as sf
# import torch
# from scipy.signal import butter, filtfilt

# from encoder import inference as encoder
# from encoder.params_model import model_embedding_size as speaker_embedding_size
# from synthesizer.inference import Synthesizer
# from utils.argutils import print_args
# from utils.default_models import ensure_default_models
# from vocoder import inference as vocoder

# import numpy as np
# if not hasattr(np, 'float'):
#     np.float = float

# import numpy as np
# if not hasattr(np, 'complex'):
#     np.complex = complex

# def normalize_audio(audio):
#     max_val = np.max(np.abs(audio))
#     if max_val > 0:
#         return audio / max_val
#     return audio

# def highpass_filter(audio, sr, cutoff=80, order=5):
#     nyquist = 0.5 * sr
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)
#     return filtfilt(b, a, audio)

# def dynamic_range_compression(audio, alpha=0.5):
#     # A simple compression: scales amplitude using a logarithmic function.
#     # Adjust alpha to change the compression strength (0 < alpha <= 1).
#     # This is a rudimentary compression and might be replaced with a more sophisticated method.
#     return np.sign(audio) * np.log1p(alpha * np.abs(audio)) / np.log1p(alpha)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument("-e", "--enc_model_fpath", type=Path,
#                         default="saved_models/default/encoder.pt",
#                         help="Path to a saved encoder")
#     parser.add_argument("-s", "--syn_model_fpath", type=Path,
#                         default="saved_models/default/synthesizer.pt",
#                         help="Path to a saved synthesizer")
#     parser.add_argument("-v", "--voc_model_fpath", type=Path,
#                         default="saved_models/default/vocoder.pt",
#                         help="Path to a saved vocoder")
#     parser.add_argument("--cpu", action="store_true", help="If True, processing is done on CPU, even when a GPU is available.")
#     parser.add_argument("--no_sound", action="store_true", help="If True, audio won't be played.")
#     parser.add_argument("--seed", type=int, default=None, help="Optional random number seed value to make toolbox deterministic.")
#     args = parser.parse_args()
#     arg_dict = vars(args)
#     print_args(args, parser)

#     # Hide GPUs from Pytorch to force CPU processing
#     if arg_dict.pop("cpu"):
#         os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#     print("Running a test of your configuration...\n")

#     if torch.cuda.is_available():
#         device_id = torch.cuda.current_device()
#         gpu_properties = torch.cuda.get_device_properties(device_id)
#         print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
#               "%.1fGb total memory.\n" %
#               (torch.cuda.device_count(),
#                device_id,
#                gpu_properties.name,
#                gpu_properties.major,
#                gpu_properties.minor,
#                gpu_properties.total_memory / 1e9))
#     else:
#         print("Using CPU for inference.\n")

#     # Load the models one by one.
#     print("Preparing the encoder, the synthesizer and the vocoder...")
#     ensure_default_models(Path("saved_models"))
#     encoder.load_model(args.enc_model_fpath)
#     synthesizer = Synthesizer(args.syn_model_fpath)
#     vocoder.load_model(args.voc_model_fpath)

#     # Run a test
#     print("Testing your configuration with small inputs.")
#     print("\tTesting the encoder...")
#     encoder.embed_utterance(np.zeros(encoder.sampling_rate))

#     embed = np.random.rand(speaker_embedding_size)
#     embed /= np.linalg.norm(embed)
#     embeds = [embed, np.zeros(speaker_embedding_size)]
#     texts = ["test 1", "test 2"]
#     print("\tTesting the synthesizer... (loading the model will output a lot of text)")
#     mels = synthesizer.synthesize_spectrograms(texts, embeds)

#     mel = np.concatenate(mels, axis=1)
#     no_action = lambda *args: None
#     print("\tTesting the vocoder...")
#     vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

#     print("All test passed! You can now synthesize speech.\n\n")

#     print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
#           "show how you can interface this project easily with your own. See the source code for "
#           "an explanation of what is happening.\n")

#     print("Interactive generation loop")
#     num_generated = 0
#     while True:
#         try:
#             message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, wav, m4a, flac, ...):\n"
#             in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))

#             # Compute the embedding
#             preprocessed_wav = encoder.preprocess_wav(in_fpath)
#             original_wav, sampling_rate = librosa.load(str(in_fpath))
#             preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
#             print("Loaded file successfully")

#             embed = encoder.embed_utterance(preprocessed_wav)
#             print("Created the embedding")

#             # Generate the spectrogram
#             text = input("Write a sentence (+-20 words) to be synthesized:\n")
#             if args.seed is not None:
#                 torch.manual_seed(args.seed)
#                 synthesizer = Synthesizer(args.syn_model_fpath)

#             texts = [text]
#             embeds = [embed]
#             specs = synthesizer.synthesize_spectrograms(texts, embeds)
#             spec = specs[0]
#             print("Created the mel spectrogram")

#             # Generate the waveform
#             print("Synthesizing the waveform:")
#             if args.seed is not None:
#                 torch.manual_seed(args.seed)
#                 vocoder.load_model(args.voc_model_fpath)

#             generated_wav = vocoder.infer_waveform(spec)

#             # Post-generation: pad and trim
#             generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
#             generated_wav = encoder.preprocess_wav(generated_wav)

#             # --- Noise Reduction Step ---
#             try:
#                 import noisereduce as nr
#                 cleaned_wav = nr.reduce_noise(y=generated_wav, sr=synthesizer.sample_rate)
#                 generated_wav = cleaned_wav
#                 print("Applied noise reduction to clean the audio.")
#             except ImportError:
#                 print("The 'noisereduce' package is not installed. Skipping noise reduction. Install it via 'pip install noisereduce'.")

#             # --- Audio Upscaling Step ---
#             # Upscale the audio from the synthesizer's sample rate to a higher rate (e.g. 48000 Hz)
#             target_sr = 48000
#             upscaled_wav = librosa.resample(generated_wav, orig_sr=synthesizer.sample_rate, target_sr=target_sr)
#             print(f"Upscaled audio from {synthesizer.sample_rate} Hz to {target_sr} Hz.")

#             # --- Additional Enhancement Methods ---

#             # 1. Normalize audio so that the maximum amplitude is 1.
#             normalized_wav = normalize_audio(upscaled_wav)
#             print("Normalized the audio.")

#             # 2. Apply a high-pass filter to remove very low-frequency noise.
#             filtered_wav = highpass_filter(normalized_wav, sr=target_sr, cutoff=80, order=5)
#             print("Applied high-pass filtering.")

#             # 3. Apply dynamic range compression to reduce volume differences.
#             compressed_wav = dynamic_range_compression(filtered_wav, alpha=0.5)
#             print("Applied dynamic range compression.")

#             enhanced_wav = compressed_wav

#             # Play the enhanced audio (non-blocking)
#             if not args.no_sound:
#                 import sounddevice as sd
#                 try:
#                     sd.stop()
#                     sd.play(enhanced_wav, target_sr)
#                 except sd.PortAudioError as e:
#                     print("\nCaught exception: %s" % repr(e))
#                     print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
#                 except Exception as e:
#                     raise e

#             # Save the final output to disk
#             filename = "demo_output_%02d.wav" % num_generated
#             sf.write(filename, enhanced_wav.astype(np.float32), target_sr)
#             num_generated += 1
#             print(f"\nSaved output as {filename}\n\n")

#         except Exception as e:
#             print("Caught exception: %s" % repr(e))
#             print("Restarting\n")



# import argparse
# import os
# from pathlib import Path

# import librosa
# import numpy as np
# import soundfile as sf
# import torch
# from scipy.signal import butter, filtfilt

# from encoder import inference as encoder
# from encoder.params_model import model_embedding_size as speaker_embedding_size
# from synthesizer.inference import Synthesizer
# from utils.argutils import print_args
# from utils.default_models import ensure_default_models
# from vocoder import inference as vocoder

# if not hasattr(np, 'float'):
#     np.float = float
# if not hasattr(np, 'complex'):
#     np.complex = complex


# def normalize_audio(audio):
#     max_val = np.max(np.abs(audio))
#     if max_val > 0:
#         return audio / max_val
#     return audio


# def highpass_filter(audio, sr, cutoff=80, order=5):
#     nyquist = 0.5 * sr
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)
#     return filtfilt(b, a, audio)


# def dynamic_range_compression(audio, alpha=0.5):
#     return np.sign(audio) * np.log1p(alpha * np.abs(audio)) / np.log1p(alpha)


# def high_quality_time_stretch(y, rate, sr, n_fft=2048, hop_length=512):
#     """
#     Time-stretch using Librosa's phase vocoder.
#     This function computes the STFT of y, applies the phase vocoder,
#     and reconstructs the audio via inverse STFT. This method preserves pitch.
#     """
#     # Compute the STFT of the input signal.
#     D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
#     # Apply the phase vocoder to stretch the time axis.
#     D_stretched = librosa.phase_vocoder(D, rate=rate, hop_length=hop_length)
#     # Reconstruct the time-domain signal from the stretched STFT.
#     y_stretched = librosa.istft(D_stretched, hop_length=hop_length)
#     return y_stretched


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument("-e", "--enc_model_fpath", type=Path,
#                         default="saved_models/default/encoder.pt",
#                         help="Path to a saved encoder")
#     parser.add_argument("-s", "--syn_model_fpath", type=Path,
#                         default="saved_models/default/synthesizer.pt",
#                         help="Path to a saved synthesizer")
#     parser.add_argument("-v", "--voc_model_fpath", type=Path,
#                         default="saved_models/default/vocoder.pt",
#                         help="Path to a saved vocoder")
#     parser.add_argument("--cpu", action="store_true", 
#                         help="If True, processing is done on CPU, even when a GPU is available.")
#     parser.add_argument("--no_sound", action="store_true", 
#                         help="If True, audio won't be played.")
#     parser.add_argument("--seed", type=int, default=None, 
#                         help="Optional random number seed value to make toolbox deterministic.")
#     # New argument to adjust tempo (stretch factor: < 1 slows audio down, > 1 speeds it up)
#     parser.add_argument("--tempo", type=float, default=0.8, 
#                         help="Tempo stretch factor. <1 slows the audio; >1 speeds it up.")
#     args = parser.parse_args()
#     arg_dict = vars(args)
#     print_args(args, parser)

#     # Hide GPUs if CPU processing is requested.
#     if arg_dict.pop("cpu"):
#         os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#     print("Running a test of your configuration...\n")
#     if torch.cuda.is_available():
#         device_id = torch.cuda.current_device()
#         gpu_properties = torch.cuda.get_device_properties(device_id)
#         print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
#               "%.1fGb total memory.\n" %
#               (torch.cuda.device_count(),
#                device_id,
#                gpu_properties.name,
#                gpu_properties.major,
#                gpu_properties.minor,
#                gpu_properties.total_memory / 1e9))
#     else:
#         print("Using CPU for inference.\n")

#     # Load the models.
#     print("Preparing the encoder, the synthesizer and the vocoder...")
#     ensure_default_models(Path("saved_models"))
#     encoder.load_model(args.enc_model_fpath)
#     synthesizer = Synthesizer(args.syn_model_fpath)
#     vocoder.load_model(args.voc_model_fpath)

#     # Run a test.
#     print("Testing your configuration with small inputs.")
#     print("\tTesting the encoder...")
#     encoder.embed_utterance(np.zeros(encoder.sampling_rate))
#     embed = np.random.rand(speaker_embedding_size)
#     embed /= np.linalg.norm(embed)
#     embeds = [embed, np.zeros(speaker_embedding_size)]
#     texts = ["test 1", "test 2"]
#     print("\tTesting the synthesizer... (loading the model will output a lot of text)")
#     mels = synthesizer.synthesize_spectrograms(texts, embeds)
#     mel = np.concatenate(mels, axis=1)
#     no_action = lambda *args: None
#     print("\tTesting the vocoder...")
#     vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
#     print("All test passed! You can now synthesize speech.\n\n")

#     print("This is a GUI-less example of interface to SV2TTS.")
#     print("Interactive generation loop")
#     num_generated = 0
#     while True:
#         try:
#             # --- Multiple Reference Audios ---
#             # Ask for multiple file paths separated by commas.
#             message = ("Reference voices: enter one or more audio filepaths (mp3, wav, m4a, flac, ...), "
#                        "separated by commas:\n")
#             ref_input = input(message).replace("\"", "").replace("\'", "")
#             # Split the input into separate paths and strip whitespace.
#             ref_paths = [Path(p.strip()) for p in ref_input.split(",") if p.strip()]

#             # Compute embeddings for each reference audio and average them.
#             embeddings = []
#             for ref_path in ref_paths:
#                 preprocessed_wav = encoder.preprocess_wav(ref_path)
#                 original_wav, sampling_rate = librosa.load(str(ref_path))
#                 preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
#                 emb = encoder.embed_utterance(preprocessed_wav)
#                 embeddings.append(emb)
#                 print(f"Processed embedding for {ref_path}")
#             if len(embeddings) == 0:
#                 raise ValueError("No valid reference audios provided.")
#             # Average the embeddings.
#             final_embed = np.mean(np.stack(embeddings), axis=0)
#             final_embed /= np.linalg.norm(final_embed)
#             print("Created the final averaged embedding.")

#             # Generate the spectrogram.
#             text = input("Write a sentence (+-20 words) to be synthesized:\n")
#             if args.seed is not None:
#                 torch.manual_seed(args.seed)
#                 synthesizer = Synthesizer(args.syn_model_fpath)
#             texts = [text]
#             embeds = [final_embed]
#             specs = synthesizer.synthesize_spectrograms(texts, embeds)
#             spec = specs[0]
#             print("Created the mel spectrogram")

#             # Generate the waveform.
#             print("Synthesizing the waveform:")
#             if args.seed is not None:
#                 torch.manual_seed(args.seed)
#                 vocoder.load_model(args.voc_model_fpath)
#             generated_wav = vocoder.infer_waveform(spec)

#             # Post-generation: pad and trim.
#             generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
#             generated_wav = encoder.preprocess_wav(generated_wav)

#             # --- Noise Reduction ---
#             try:
#                 import noisereduce as nr
#                 cleaned_wav = nr.reduce_noise(y=generated_wav, sr=synthesizer.sample_rate)
#                 generated_wav = cleaned_wav
#                 print("Applied noise reduction.")
#             except ImportError:
#                 print("The 'noisereduce' package is not installed. Skipping noise reduction.")

#             # --- Audio Upscaling ---
#             target_sr = 48000
#             upscaled_wav = librosa.resample(generated_wav, orig_sr=synthesizer.sample_rate, target_sr=target_sr)
#             print(f"Upscaled audio from {synthesizer.sample_rate} Hz to {target_sr} Hz.")

#             # --- Additional Enhancements ---
#             normalized_wav = normalize_audio(upscaled_wav)
#             filtered_wav = highpass_filter(normalized_wav, sr=target_sr, cutoff=80, order=5)
#             compressed_wav = dynamic_range_compression(filtered_wav, alpha=0.5)
#             enhanced_wav = compressed_wav

#             # --- Reduce the Tempo while Preserving Pitch ---
#             # The tempo factor (<1 slows audio down, >1 speeds it up)
#             tempo_factor = args.tempo
#             # Use the custom high-quality time stretching function.
#             tempo_adjusted_wav = high_quality_time_stretch(enhanced_wav, rate=tempo_factor, sr=target_sr)
#             print(f"Applied time stretching with a factor of {tempo_factor}, preserving pitch.")

#             # Play the enhanced audio (non-blocking)
#             if not args.no_sound:
#                 import sounddevice as sd
#                 try:
#                     sd.stop()
#                     sd.play(tempo_adjusted_wav, target_sr)
#                 except sd.PortAudioError as e:
#                     print("\nCaught exception: %s" % repr(e))
#                     print("Continuing without audio playback. Use the \"--no_sound\" flag to suppress this message.\n")
#                 except Exception as e:
#                     raise e

#             # Save the final output.
#             filename = "demo_output_%02d.wav" % num_generated
#             sf.write(filename, tempo_adjusted_wav.astype(np.float32), target_sr)
#             num_generated += 1
#             print(f"\nSaved output as {filename}\n\n")

#         except Exception as e:
#             print("Caught exception: %s" % repr(e))
#             print("Restarting\n")

# import argparse
# import os
# from pathlib import Path

# import librosa
# import numpy as np
# import soundfile as sf
# import torch
# from scipy.signal import butter, filtfilt, medfilt

# from encoder import inference as encoder
# from encoder.params_model import model_embedding_size as speaker_embedding_size
# from synthesizer.inference import Synthesizer
# from utils.argutils import print_args
# from utils.default_models import ensure_default_models
# from vocoder import inference as vocoder

# if not hasattr(np, 'float'):
#     np.float = float
# if not hasattr(np, 'complex'):
#     np.complex = complex


# def normalize_audio(audio):
#     max_val = np.max(np.abs(audio))
#     if max_val > 0:
#         return audio / max_val
#     return audio


# def highpass_filter(audio, sr, cutoff=80, order=5):
#     nyquist = 0.5 * sr
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)
#     return filtfilt(b, a, audio)


# def dynamic_range_compression(audio, alpha=0.4):
#     """Apply a gentler compression to preserve natural dynamics."""
#     return np.sign(audio) * np.log1p(alpha * np.abs(audio)) / np.log1p(alpha)


# def high_quality_time_stretch(y, rate, sr, n_fft=4096, hop_length=512):
#     """
#     Time-stretch using Librosa's phase vocoder.
#     Increasing n_fft and hop_length can smooth transitions and reduce artifacts.
#     """
#     # Compute the STFT of the input signal.
#     D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
#     # Apply the phase vocoder to adjust the time scale.
#     D_stretched = librosa.phase_vocoder(D, rate=rate, hop_length=hop_length)
#     # Reconstruct the time-domain signal from the stretched STFT.
#     y_stretched = librosa.istft(D_stretched, hop_length=hop_length)
#     # Apply a mild median filter to smooth out any abrupt artifacts.
#     y_stretched = medfilt(y_stretched, kernel_size=3)
#     return y_stretched


# def enhance_tone(audio, sr, cutoff=3000, gain=1.1, n_fft=2048, hop_length=512):
#     """
#     Enhance tone and sharpness by boosting frequencies above a cutoff.
#     Frequencies above 'cutoff' Hz are boosted by the specified gain.
#     """
#     # Compute the STFT.
#     D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
#     # Get the frequency bins.
#     freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#     # Create a boost factor: 1 for frequencies below cutoff, 'gain' for frequencies above.
#     boost = np.where(freqs >= cutoff, gain, 1.0)
#     # Reshape boost vector for broadcasting.
#     boost = boost[:, None]
#     # Apply the boost.
#     D_enhanced = D * boost
#     # Reconstruct the enhanced audio.
#     audio_enhanced = librosa.istft(D_enhanced, hop_length=hop_length)
#     return audio_enhanced


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument("-e", "--enc_model_fpath", type=Path,
#                         default="saved_models/default/encoder.pt",
#                         help="Path to a saved encoder")
#     parser.add_argument("-s", "--syn_model_fpath", type=Path,
#                         default="saved_models/default/synthesizer.pt",
#                         help="Path to a saved synthesizer")
#     parser.add_argument("-v", "--voc_model_fpath", type=Path,
#                         default="saved_models/default/vocoder.pt",
#                         help="Path to a saved vocoder")
#     parser.add_argument("--cpu", action="store_true", 
#                         help="If True, processing is done on CPU, even when a GPU is available.")
#     parser.add_argument("--no_sound", action="store_true", 
#                         help="If True, audio won't be played.")
#     parser.add_argument("--seed", type=int, default=None, 
#                         help="Optional random number seed value to make toolbox deterministic.")
#     # Tempo stretch factor: < 1 slows audio; > 1 speeds it up.
#     parser.add_argument("--tempo", type=float, default=0.8, 
#                         help="Tempo stretch factor. <1 slows the audio; >1 speeds it up.")
#     args = parser.parse_args()
#     arg_dict = vars(args)
#     print_args(args, parser)

#     if arg_dict.pop("cpu"):
#         os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#     print("Running a test of your configuration...\n")
#     if torch.cuda.is_available():
#         device_id = torch.cuda.current_device()
#         gpu_properties = torch.cuda.get_device_properties(device_id)
#         print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
#               "%.1fGb total memory.\n" %
#               (torch.cuda.device_count(),
#                device_id,
#                gpu_properties.name,
#                gpu_properties.major,
#                gpu_properties.minor,
#                gpu_properties.total_memory / 1e9))
#     else:
#         print("Using CPU for inference.\n")

#     print("Preparing the encoder, the synthesizer and the vocoder...")
#     ensure_default_models(Path("saved_models"))
#     encoder.load_model(args.enc_model_fpath)
#     synthesizer = Synthesizer(args.syn_model_fpath)
#     vocoder.load_model(args.voc_model_fpath)

#     print("Testing your configuration with small inputs.")
#     print("\tTesting the encoder...")
#     encoder.embed_utterance(np.zeros(encoder.sampling_rate))
#     embed = np.random.rand(speaker_embedding_size)
#     embed /= np.linalg.norm(embed)
#     embeds = [embed, np.zeros(speaker_embedding_size)]
#     texts = ["test 1", "test 2"]
#     print("\tTesting the synthesizer... (loading the model will output a lot of text)")
#     mels = synthesizer.synthesize_spectrograms(texts, embeds)
#     mel = np.concatenate(mels, axis=1)
#     no_action = lambda *args: None
#     print("\tTesting the vocoder...")
#     vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
#     print("All test passed! You can now synthesize speech.\n\n")

#     print("This is a GUI-less example of interface to SV2TTS.")
#     print("Interactive generation loop")
#     num_generated = 0
#     while True:
#         try:
#             message = ("Reference voices: enter one or more audio filepaths (mp3, wav, m4a, flac, ...), "
#                        "separated by commas:\n")
#             ref_input = input(message).replace("\"", "").replace("\'", "")
#             ref_paths = [Path(p.strip()) for p in ref_input.split(",") if p.strip()]

#             # Compute embeddings for each reference audio and average them.
#             embeddings = []
#             for ref_path in ref_paths:
#                 preprocessed_wav = encoder.preprocess_wav(ref_path)
#                 original_wav, sampling_rate = librosa.load(str(ref_path))
#                 preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
#                 emb = encoder.embed_utterance(preprocessed_wav)
#                 embeddings.append(emb)
#                 print(f"Processed embedding for {ref_path}")
#             if len(embeddings) == 0:
#                 raise ValueError("No valid reference audios provided.")
#             final_embed = np.mean(np.stack(embeddings), axis=0)
#             final_embed /= np.linalg.norm(final_embed)
#             print("Created the final averaged embedding.")

#             text = input("Write a sentence (+-20 words) to be synthesized:\n")
#             if args.seed is not None:
#                 torch.manual_seed(args.seed)
#                 synthesizer = Synthesizer(args.syn_model_fpath)
#             texts = [text]
#             embeds = [final_embed]
#             specs = synthesizer.synthesize_spectrograms(texts, embeds)
#             spec = specs[0]
#             print("Created the mel spectrogram")

#             print("Synthesizing the waveform:")
#             if args.seed is not None:
#                 torch.manual_seed(args.seed)
#                 vocoder.load_model(args.voc_model_fpath)
#             generated_wav = vocoder.infer_waveform(spec)

#             generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
#             generated_wav = encoder.preprocess_wav(generated_wav)

#             try:
#                 import noisereduce as nr
#                 cleaned_wav = nr.reduce_noise(y=generated_wav, sr=synthesizer.sample_rate)
#                 generated_wav = cleaned_wav
#                 print("Applied initial noise reduction.")
#             except ImportError:
#                 print("The 'noisereduce' package is not installed. Skipping initial noise reduction.")

#             target_sr = 48000
#             upscaled_wav = librosa.resample(generated_wav, orig_sr=synthesizer.sample_rate, target_sr=target_sr)
#             print(f"Upscaled audio from {synthesizer.sample_rate} Hz to {target_sr} Hz.")

#             normalized_wav = normalize_audio(upscaled_wav)
#             filtered_wav = highpass_filter(normalized_wav, sr=target_sr, cutoff=80, order=5)
#             compressed_wav = dynamic_range_compression(filtered_wav, alpha=0.4)
#             enhanced_wav = compressed_wav

#             tempo_factor = args.tempo
#             tempo_adjusted_wav = high_quality_time_stretch(enhanced_wav, rate=tempo_factor, sr=target_sr)
#             print(f"Applied time stretching with a factor of {tempo_factor}, preserving pitch.")

#             tone_enhanced_wav = enhance_tone(tempo_adjusted_wav, sr=target_sr, cutoff=3000, gain=1.1)
#             print("Enhanced tone and sharpness by boosting high frequencies.")

#             try:
#                 import noisereduce as nr
#                 final_wav = nr.reduce_noise(y=tone_enhanced_wav, sr=target_sr, prop_decrease=0.5, time_constant_s=0.4)
#                 print("Applied final noise reduction (gentle settings).")
#             except ImportError:
#                 final_wav = tone_enhanced_wav
#                 print("The 'noisereduce' package is not installed. Skipping final noise reduction.")

#             if not args.no_sound:
#                 import sounddevice as sd
#                 try:
#                     sd.stop()
#                     sd.play(final_wav, target_sr)
#                 except sd.PortAudioError as e:
#                     print("\nCaught exception: %s" % repr(e))
#                     print("Continuing without audio playback. Use the \"--no_sound\" flag to suppress this message.\n")
#                 except Exception as e:
#                     raise e

#             filename = "demo_output_%02d.wav" % num_generated
#             sf.write(filename, final_wav.astype(np.float32), target_sr)
#             num_generated += 1
#             print(f"\nSaved output as {filename}\n\n")

#         except Exception as e:
#             print("Caught exception: %s" % repr(e))
#             print("Restarting\n")

# import argparse
# import os
# from pathlib import Path

# import librosa
# import numpy as np
# import soundfile as sf
# import torch
# from scipy.signal import butter, filtfilt, medfilt

# from encoder import inference as encoder
# from encoder.params_model import model_embedding_size as speaker_embedding_size
# from synthesizer.inference import Synthesizer
# from utils.argutils import print_args
# from utils.default_models import ensure_default_models
# from vocoder import inference as vocoder

# if not hasattr(np, 'float'):
#     np.float = float
# if not hasattr(np, 'complex'):
#     np.complex = complex


# def normalize_audio(audio):
#     max_val = np.max(np.abs(audio))
#     return audio / max_val if max_val > 0 else audio


# def highpass_filter(audio, sr, cutoff=80, order=5):
#     nyquist = 0.5 * sr
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)
#     return filtfilt(b, a, audio)


# def dynamic_range_compression(audio, alpha=0.4):
#     """Apply a gentle compression to preserve natural dynamics."""
#     return np.sign(audio) * np.log1p(alpha * np.abs(audio)) / np.log1p(alpha)


# def high_quality_time_stretch(y, rate, sr, n_fft=4096, hop_length=512):
#     """
#     Time-stretch using Librosa's phase vocoder.
#     Increasing n_fft and hop_length can smooth transitions and reduce artifacts.
#     """
#     D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
#     D_stretched = librosa.phase_vocoder(D, rate=rate, hop_length=hop_length)
#     y_stretched = librosa.istft(D_stretched, hop_length=hop_length)
#     y_stretched = medfilt(y_stretched, kernel_size=3)
#     return y_stretched


# def enhance_tone(audio, sr, cutoff=3000, gain=1.2, n_fft=2048, hop_length=512):
#     """
#     Enhance tone and sharpness by boosting frequencies above a cutoff.
#     Frequencies above 'cutoff' Hz are boosted by the specified gain.
#     """
#     D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
#     freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#     boost = np.where(freqs >= cutoff, gain, 1.0)[:, None]
#     D_enhanced = D * boost
#     audio_enhanced = librosa.istft(D_enhanced, hop_length=hop_length)
#     return audio_enhanced


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument("-e", "--enc_model_fpath", type=Path,
#                         default="saved_models/default/encoder.pt",
#                         help="Path to a saved encoder")
#     parser.add_argument("-s", "--syn_model_fpath", type=Path,
#                         default="saved_models/default/synthesizer.pt",
#                         help="Path to a saved synthesizer")
#     parser.add_argument("-v", "--voc_model_fpath", type=Path,
#                         default="saved_models/default/vocoder.pt",
#                         help="Path to a saved vocoder")
#     parser.add_argument("--cpu", action="store_true",
#                         help="If True, processing is done on CPU, even when a GPU is available.")
#     parser.add_argument("--no_sound", action="store_true",
#                         help="If True, audio won't be played.")
#     parser.add_argument("--seed", type=int, default=42,
#                         help="Optional random number seed value to make toolbox deterministic.")
#     # Tempo stretch factor: < 1 slows audio; > 1 speeds it up.
#     parser.add_argument("--tempo", type=float, default=1.05,
#                         help="Tempo stretch factor. <1 slows the audio; >1 speeds it up.")
#     args = parser.parse_args()
#     arg_dict = vars(args)
#     print_args(args, parser)

#     if arg_dict.pop("cpu"):
#         os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#     print("Running a test of your configuration...\n")
#     if torch.cuda.is_available():
#         device_id = torch.cuda.current_device()
#         gpu_properties = torch.cuda.get_device_properties(device_id)
#         print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
#               "%.1fGb total memory.\n" %
#               (torch.cuda.device_count(),
#                device_id,
#                gpu_properties.name,
#                gpu_properties.major,
#                gpu_properties.minor,
#                gpu_properties.total_memory / 1e9))
#     else:
#         print("Using CPU for inference.\n")

#     print("Preparing the encoder, the synthesizer and the vocoder...")
#     ensure_default_models(Path("saved_models"))
#     encoder.load_model(args.enc_model_fpath)
#     synthesizer = Synthesizer(args.syn_model_fpath)
#     vocoder.load_model(args.voc_model_fpath)

#     print("Testing your configuration with small inputs.")
#     print("\tTesting the encoder...")
#     encoder.embed_utterance(np.zeros(encoder.sampling_rate))
#     embed = np.random.rand(speaker_embedding_size)
#     embed /= np.linalg.norm(embed)
#     embeds = [embed, np.zeros(speaker_embedding_size)]
#     texts = ["test 1", "test 2"]
#     print("\tTesting the synthesizer... (loading the model will output a lot of text)")
#     mels = synthesizer.synthesize_spectrograms(texts, embeds)
#     mel = np.concatenate(mels, axis=1)
#     no_action = lambda *args: None
#     print("\tTesting the vocoder...")
#     vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
#     print("All test passed! You can now synthesize speech.\n\n")

#     print("This is a GUI-less example of interface to SV2TTS.")
#     print("Interactive generation loop")
#     num_generated = 0
#     while True:
#         try:
#             message = ("Reference voices: enter one or more audio filepaths (mp3, wav, m4a, flac, ...), "
#                        "separated by commas:\n")
#             ref_input = input(message).replace("\"", "").replace("\'", "")
#             ref_paths = [Path(p.strip()) for p in ref_input.split(",") if p.strip()]

#             # Compute embeddings for each reference audio and average them.
#             embeddings = []
#             for ref_path in ref_paths:
#                 preprocessed_wav = encoder.preprocess_wav(ref_path)
#                 original_wav, sampling_rate = librosa.load(str(ref_path))
#                 preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
#                 emb = encoder.embed_utterance(preprocessed_wav)
#                 embeddings.append(emb)
#                 print(f"Processed embedding for {ref_path}")
#             if len(embeddings) == 0:
#                 raise ValueError("No valid reference audios provided.")
#             final_embed = np.mean(np.stack(embeddings), axis=0)
#             final_embed /= np.linalg.norm(final_embed)
#             print("Created the final averaged embedding.")

#             text = input("Write a sentence (+-20 words) to be synthesized:\n")
#             if args.seed is not None:
#                 torch.manual_seed(args.seed)
#                 synthesizer = Synthesizer(args.syn_model_fpath)
#             texts = [text]
#             embeds = [final_embed]
#             specs = synthesizer.synthesize_spectrograms(texts, embeds)
#             spec = specs[0]
#             print("Created the mel spectrogram")

#             print("Synthesizing the waveform:")
#             if args.seed is not None:
#                 torch.manual_seed(args.seed)
#                 vocoder.load_model(args.voc_model_fpath)
#             generated_wav = vocoder.infer_waveform(spec)

#             generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
#             generated_wav = encoder.preprocess_wav(generated_wav)

#             # --- Initial Noise Reduction ---
#             try:
#                 import noisereduce as nr
#                 cleaned_wav = nr.reduce_noise(y=generated_wav, sr=synthesizer.sample_rate)
#                 generated_wav = cleaned_wav
#                 print("Applied initial noise reduction.")
#             except ImportError:
#                 print("The 'noisereduce' package is not installed. Skipping initial noise reduction.")

#             # --- Audio Upscaling ---
#             target_sr = 48000
#             upscaled_wav = librosa.resample(generated_wav, orig_sr=synthesizer.sample_rate, target_sr=target_sr)
#             print(f"Upscaled audio from {synthesizer.sample_rate} Hz to {target_sr} Hz.")

#             normalized_wav = normalize_audio(upscaled_wav)
#             filtered_wav = highpass_filter(normalized_wav, sr=target_sr, cutoff=80, order=5)
#             compressed_wav = dynamic_range_compression(filtered_wav, alpha=0.4)
#             enhanced_wav = compressed_wav

#             # --- Time Stretching while Preserving Pitch ---
#             tempo_factor = args.tempo
#             tempo_adjusted_wav = high_quality_time_stretch(enhanced_wav, rate=tempo_factor, sr=target_sr)
#             print(f"Applied time stretching with a factor of {tempo_factor}, preserving pitch.")

#             # --- Tone and Sharpness Enhancement ---
#             tone_enhanced_wav = enhance_tone(tempo_adjusted_wav, sr=target_sr, cutoff=3000, gain=1.2)
#             print("Enhanced tone and sharpness by boosting high frequencies.")

#             # Final output (without additional Wiener filtering).
#             final_wav = tone_enhanced_wav

#             if not args.no_sound:
#                 import sounddevice as sd
#                 try:
#                     sd.stop()
#                     sd.play(final_wav, target_sr)
#                 except sd.PortAudioError as e:
#                     print("\nCaught exception: %s" % repr(e))
#                     print("Continuing without audio playback. Use the \"--no_sound\" flag to suppress this message.\n")
#                 except Exception as e:
#                     raise e

#             filename = "demo_output_%02d.wav" % num_generated
#             sf.write(filename, final_wav.astype(np.float32), target_sr)
#             num_generated += 1
#             print(f"\nSaved output as {filename}\n\n")

#         except Exception as e:
#             print("Caught exception: %s" % repr(e))
#             print("Restarting\n")

# Final checkpoint

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, filtfilt, medfilt

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex


def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 0 else audio


def highpass_filter(audio, sr, cutoff=80, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, audio)


def dynamic_range_compression(audio, alpha=0.4):
    """Apply a gentle compression to preserve natural dynamics."""
    return np.sign(audio) * np.log1p(alpha * np.abs(audio)) / np.log1p(alpha)


def high_quality_time_stretch(y, rate, sr, n_fft=4096, hop_length=512):
    """
    Time-stretch using Librosa's phase vocoder.
    Increasing n_fft and hop_length can smooth transitions and reduce artifacts.
    """
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    D_stretched = librosa.phase_vocoder(D, rate=rate, hop_length=hop_length)
    y_stretched = librosa.istft(D_stretched, hop_length=hop_length)
    y_stretched = medfilt(y_stretched, kernel_size=3)
    return y_stretched


def enhance_tone(audio, sr, cutoff=3000, gain=1.2, n_fft=2048, hop_length=512):
    """
    Enhance tone and sharpness by boosting frequencies above a cutoff.
    Frequencies above 'cutoff' Hz are boosted by the specified gain.
    """
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    boost = np.where(freqs >= cutoff, gain, 1.0)[:, None]
    D_enhanced = D * boost
    audio_enhanced = librosa.istft(D_enhanced, hop_length=hop_length)
    return audio_enhanced


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true",
                        help="If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true",
                        help="If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Optional random number seed value to make toolbox deterministic.")
    # Tempo stretch factor: < 1 slows audio; > 1 speeds it up.
    parser.add_argument("--tempo", type=float, default=1,
                        help="Tempo stretch factor. <1 slows the audio; >1 speeds it up.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
              "%.1fGb total memory.\n" %
              (torch.cuda.device_count(),
               device_id,
               gpu_properties.name,
               gpu_properties.major,
               gpu_properties.minor,
               gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    print("Preparing the encoder, the synthesizer and the vocoder...")
    ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)

    print("Testing your configuration with small inputs.")
    print("\tTesting the encoder...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    embed = np.random.rand(speaker_embedding_size)
    embed /= np.linalg.norm(embed)
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ["test 1", "test 2"]
    print("\tTesting the synthesizer... (loading the model will output a lot of text)")
    mels = synthesizer.synthesize_spectrograms(texts, embeds)
    mel = np.concatenate(mels, axis=1)
    no_action = lambda *args: None
    print("\tTesting the vocoder...")
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
    print("All test passed! You can now synthesize speech.\n\n")

    print("This is a GUI-less example of interface to SV2TTS.")
    print("Interactive generation loop")
    num_generated = 0
    while True:
        try:
            message = ("Reference voices: enter one or more audio filepaths (mp3, wav, m4a, flac, ...), "
                       "separated by commas:\n")
            ref_input = input(message).replace("\"", "").replace("\'", "")
            ref_paths = [Path(p.strip()) for p in ref_input.split(",") if p.strip()]

            # Compute embeddings for each reference audio and average them.
            embeddings = []
            for ref_path in ref_paths:
                preprocessed_wav = encoder.preprocess_wav(ref_path)
                original_wav, sampling_rate = librosa.load(str(ref_path))
                preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
                emb = encoder.embed_utterance(preprocessed_wav)
                embeddings.append(emb)
                print(f"Processed embedding for {ref_path}")
            if len(embeddings) == 0:
                raise ValueError("No valid reference audios provided.")
            final_embed = np.mean(np.stack(embeddings), axis=0)
            final_embed /= np.linalg.norm(final_embed)
            print("Created the final averaged embedding.")

            text = input("Write a sentence (+-20 words) to be synthesized:\n")
            if args.seed is not None:
                torch.manual_seed(args.seed)
                synthesizer = Synthesizer(args.syn_model_fpath)
            texts = [text]
            embeds = [final_embed]
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")

            print("Synthesizing the waveform:")
            if args.seed is not None:
                torch.manual_seed(args.seed)
                vocoder.load_model(args.voc_model_fpath)
            generated_wav = vocoder.infer_waveform(spec)

            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
            generated_wav = encoder.preprocess_wav(generated_wav)

            # --- Initial Noise Reduction ---
            try:
                import noisereduce as nr
                cleaned_wav = nr.reduce_noise(y=generated_wav, sr=synthesizer.sample_rate)
                generated_wav = cleaned_wav
                print("Applied initial noise reduction.")
            except ImportError:
                print("The 'noisereduce' package is not installed. Skipping initial noise reduction.")

            # --- Audio Upscaling ---
            # Upscale to 64 kHz to add more presence.
            target_sr = 64000
            upscaled_wav = librosa.resample(generated_wav, orig_sr=synthesizer.sample_rate, target_sr=target_sr)
            print(f"Upscaled audio from {synthesizer.sample_rate} Hz to {target_sr} Hz.")

            normalized_wav = normalize_audio(upscaled_wav)
            filtered_wav = highpass_filter(normalized_wav, sr=target_sr, cutoff=80, order=5)
            compressed_wav = dynamic_range_compression(filtered_wav, alpha=0.4)
            enhanced_wav = compressed_wav

            # --- Time Stretching while Preserving Pitch ---
            tempo_factor = args.tempo
            tempo_adjusted_wav = high_quality_time_stretch(enhanced_wav, rate=tempo_factor, sr=target_sr)
            print(f"Applied time stretching with a factor of {tempo_factor}, preserving pitch.")

            # --- Tone and Sharpness Enhancement ---
            tone_enhanced_wav = enhance_tone(tempo_adjusted_wav, sr=target_sr, cutoff=3000, gain=1.2)
            print("Enhanced tone and sharpness by boosting high frequencies.")

            # --- Apply Final Gain Boost ---
            # Boost overall amplitude to bring the voice closer.
            final_gain = 2.0
            final_wav = normalize_audio(tone_enhanced_wav * final_gain)
            print(f"Applied final gain boost of {final_gain}.")

            if not args.no_sound:
                import sounddevice as sd
                try:
                    sd.stop()
                    sd.play(final_wav, target_sr)
                except sd.PortAudioError as e:
                    print("\nCaught exception: %s" % repr(e))
                    print("Continuing without audio playback. Use the \"--no_sound\" flag to suppress this message.\n")
                except Exception as e:
                    raise e

            filename = "demo_output_%02d.wav" % num_generated
            sf.write(filename, final_wav.astype(np.float32), target_sr)
            num_generated += 1
            print(f"\nSaved output as {filename}\n\n")

        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
