# V0 Prototype for extracting haptic events from audio.

import numpy as np
import librosa
import scipy.signal
import json

# Load the audio
audio_path = "audio.wav"
y, sr = librosa.load(audio_path, sr=44100)
frame_length = 2048
hop_length = 512
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

# Detect peaks in the energy signal
# You can adjust 'height' and 'distance' to be more/less sensitive
peaks, _ = scipy.signal.find_peaks(rms, height=0.17, distance=10)  # Play with height value if needed

# Prepare haptic events
haptic_events = []
for peak in peaks:
    event = {
        "time": float(times[peak]),
        "type": "transient",      # Short, strong pulse for now
        "intensity": float(rms[peak])  # We'll map this to haptic intensity later
    }
    haptic_events.append(event)

# Save to JSON
with open("haptic_events.json", "w") as f:
    json.dump(haptic_events, f, indent=2)

print(f"Detected {len(haptic_events)} haptic events. Saved to haptic_events.json")
