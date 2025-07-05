import librosa
import numpy as np
import json

# ---- CONFIG ----
audio_file = 'audio.wav'
yamnet_file = 'yamnet_timeline.json'
output_ahap = 'hybrid.ahap'

frame_length = 2048    # ~50ms at 44.1kHz
hop_length = 1024      # ~25ms

MASK_LABELS = ['Speech', 'Silence']  # Mask these

# ---- 1. Load audio and extract features ----
y, sr = librosa.load(audio_file, sr=44100, mono=True)

rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length, n_fft=frame_length)

# Normalize to [0, 1]
intensity = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
sharpness = (cent - cent.min()) / (cent.max() - cent.min() + 1e-6)

# ---- 2. Load YAMNet label timeline ----
with open(yamnet_file) as f:
    yamnet = json.load(f)

# Build a lookup of all intervals to mask (Speech, Music, Silence, etc.)
mask_intervals = []
win_sec = yamnet[1]['time'] - yamnet[0]['time'] if len(yamnet) > 1 else 1.0
for entry in yamnet:
    if entry['label'] in MASK_LABELS:
        t0 = entry['time']
        t1 = t0 + win_sec
        mask_intervals.append((t0, t1))

def is_masked(t):
    # Is time t inside any masked interval?
    for t0, t1 in mask_intervals:
        if t0 <= t < t1:
            return True
    return False

# ---- 3. Create haptics only for non-masked frames ----
ahap_events = []
for t, inten, sharp in zip(times, intensity, sharpness):
    if is_masked(t):
        continue  # skip masked labels
    if inten > 0.07:
        ahap_events.append({
            "Event": {
                "Time": round(float(t), 3),
                "EventType": "HapticTransient",
                "EventParameters": [
                    {"ParameterID": "HapticIntensity", "ParameterValue": float(inten)},
                    {"ParameterID": "HapticSharpness", "ParameterValue": float(sharp)}
                ]
            }
        })

ahap = {
    "Version": 1,
    "Pattern": ahap_events
}

with open(output_ahap, "w") as f:
    json.dump(ahap, f, indent=2)

print(f"Generated {output_ahap} with {len(ahap_events)} haptic events.")
