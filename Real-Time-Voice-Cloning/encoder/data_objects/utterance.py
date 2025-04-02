import numpy as np
import librosa
from encoder.params_data import *  # Ensure that sampling_rate, mel_window_length, mel_window_step, and mel_n_channels are defined
from pathlib import Path

class Utterance:
    def __init__(self, frames_fpath: Path, wave_fpath: Path):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath
        
    def get_frames(self):
        # If the frames file exists and is a .npy file, load it
        if self.frames_fpath.exists() and self.frames_fpath.suffix == ".npy":
            try:
                return np.load(self.frames_fpath, allow_pickle=True)
            except Exception as e:
                print(f"Error loading frames from {self.frames_fpath}: {e}")
                # Fall back to computing frames if loading fails.
        # Otherwise, compute the mel-spectrogram from the wav file.
        wav, sr = librosa.load(self.wave_fpath, sr=sampling_rate)
        frames = librosa.feature.melspectrogram(
            y=wav,
            sr=sampling_rate,
            n_fft=int(sampling_rate * mel_window_length / 1000),
            hop_length=int(sampling_rate * mel_window_step / 1000),
            n_mels=mel_n_channels
        )
        frames = frames.astype(np.float32).T
        # Optionally, save the computed frames for future use:
        # save_path = self.frames_fpath.with_suffix(".npy")
        # np.save(save_path, frames, allow_pickle=True)
        return frames

    def random_partial(self, n_frames):
        frames = self.get_frames()
        total_frames = frames.shape[0]
        if total_frames < n_frames:
            # Pad the frames with zeros along the time dimension
            pad_width = n_frames - total_frames
            frames = np.pad(frames, ((0, pad_width), (0, 0)), mode='constant')
            start = 0
        else:
            if total_frames == n_frames:
                start = 0
            else:
                start = np.random.randint(0, total_frames - n_frames + 1)
        end = start + n_frames
        return frames[start:end], (start, end)

