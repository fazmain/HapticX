import librosa
import numpy as np
import json

audio_file = 'audio.wav'
y, sr = librosa.load(audio_file, sr=44100, mono=True)

frame_length = 2048  # ~50ms
hop_length = 1024    # ~25ms

# Compute root-mean-square energy (loudness)
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

# Compute spectral centroid (brightness)
cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length, n_fft=frame_length)

# Normalize to [0, 1]
intensity = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
sharpness = (cent - cent.min()) / (cent.max() - cent.min() + 1e-6)

ahap_events = []
for t, inten, sharp in zip(times, intensity, sharpness):
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

with open("pattern_generated.ahap", "w") as f:
    json.dump(ahap, f, indent=2)

print(f"Generated pattern_generated.ahap with {len(ahap_events)} events.")
