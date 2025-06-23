import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load the audio file
audio_path = "audio.wav"
y, sr = librosa.load(audio_path, sr=44100)

# Compute short-term energy (root mean square - RMS)
frame_length = 2048
hop_length = 512
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

# Plot audio waveform and RMS energy
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.title('Waveform')
plt.subplot(2, 1, 2)
plt.plot(times, rms, color='red')
plt.title('Short-Term Energy (RMS)')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.tight_layout()
plt.show()
