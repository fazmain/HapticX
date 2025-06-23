import librosa
import numpy as np
import json

# --- Config ---
audio_file = 'audio.wav'
yamnet_file = 'yamnet_timeline.json'
output_ahap = 'pattern_hybrid.ahap'

frame_length = 2048     # ~50ms (can be longer for YAMNet alignment, e.g. 1s)
hop_length = 1024       # ~25ms (or 0.5s if matching YAMNet hop)

# --- Special Events Mapping ---
EVENT_CATEGORIES = {
    "Explosion": "Explosion",
    "Boom": "Explosion",
    "Eruption": "Explosion",
    "Gunshot, gunfire": "Gunfire",
    "Artillery fire": "Gunfire",
    "Music": "Music",
    "Vehicle": "Engine",
    "Aircraft": "Engine",
    "Boat, Water vehicle": "Engine",
    "Whoosh, swoosh, swish": "Whoosh",
    "Wind": "Whoosh"
}
MASK_CLASSES = ["Speech", "Silence", "Water", "Crowd"]

# --- Helper Functions for Event Patterns ---
def make_explosion(time):
    return [{
        "Event": {
            "Time": round(time, 3),
            "EventType": "HapticTransient",
            "EventParameters": [
                {"ParameterID": "HapticIntensity", "ParameterValue": 1.0},
                {"ParameterID": "HapticSharpness", "ParameterValue": 1.0}
            ]
        }
    }]

def make_gunfire(time):
    return [{
        "Event": {
            "Time": round(time, 3),
            "EventType": "HapticTransient",
            "EventParameters": [
                {"ParameterID": "HapticIntensity", "ParameterValue": 0.8},
                {"ParameterID": "HapticSharpness", "ParameterValue": 1.0}
            ]
        }
    }]

def make_music(time, duration=1.5):
    # "Inflate"-style pattern
    return [
        {
            "Event": {
                "Time": round(time, 3),
                "EventType": "HapticContinuous",
                "EventDuration": duration,
                "EventParameters": [
                    {"ParameterID": "HapticIntensity", "ParameterValue": 1.0},
                    {"ParameterID": "HapticSharpness", "ParameterValue": 0.5}
                ]
            }
        },
        {
            "ParameterCurve": {
                "ParameterID": "HapticIntensityControl",
                "Time": round(time, 3),
                "ParameterCurveControlPoints": [
                    { "Time": round(time, 3), "ParameterValue": 0.0 },
                    { "Time": round(time + 1.1, 3), "ParameterValue": 0.5 },
                    { "Time": round(time + duration, 3), "ParameterValue": 0.0 }
                ]
            }
        },
        {
            "ParameterCurve": {
                "ParameterID": "HapticSharpnessControl",
                "Time": round(time, 3),
                "ParameterCurveControlPoints": [
                    { "Time": round(time, 3), "ParameterValue": -0.8 },
                    { "Time": round(time + duration, 3), "ParameterValue": 0.8 }
                ]
            }
        }
    ]

def make_engine(time, duration=1.0, y=None, sr=None):
    # Feature-driven rumble: modulate intensity/sharpness from the local engine chunk
    events = []
    if y is not None and sr is not None:
        start_sample = int(time * sr)
        end_sample = int((time + duration) * sr)
        y_chunk = y[start_sample:end_sample]
        chunk_frame = int(0.1 * sr)
        chunk_hop = chunk_frame
        rms = librosa.feature.rms(y=y_chunk, frame_length=chunk_frame, hop_length=chunk_hop)[0]
        cent = librosa.feature.spectral_centroid(y=y_chunk, sr=sr, n_fft=chunk_frame, hop_length=chunk_hop)[0]
        n_frames = len(rms)
        for i in range(n_frames):
            t = round(time + i * 0.1, 3)
            inten = float((rms[i] - rms.min()) / (rms.max() - rms.min() + 1e-6))
            sharp = float((cent[i] - cent.min()) / (cent.max() - cent.min() + 1e-6))
            events.append({
                "Event": {
                    "Time": t,
                    "EventType": "HapticTransient",
                    "EventParameters": [
                        {"ParameterID": "HapticIntensity", "ParameterValue": inten},
                        {"ParameterID": "HapticSharpness", "ParameterValue": sharp}
                    ]
                }
            })
    else:
        # fallback: simple rumble
        for i in range(10):
            t = round(time + i * 0.1, 3)
            events.append({
                "Event": {
                    "Time": t,
                    "EventType": "HapticTransient",
                    "EventParameters": [
                        {"ParameterID": "HapticIntensity", "ParameterValue": 1.0},
                        {"ParameterID": "HapticSharpness", "ParameterValue": 0.1}
                    ]
                }
            })
    return events

def make_whoosh(time):
    return [{
        "Event": {
            "Time": round(time, 3),
            "EventType": "HapticTransient",
            "EventParameters": [
                {"ParameterID": "HapticIntensity", "ParameterValue": 0.7},
                {"ParameterID": "HapticSharpness", "ParameterValue": 0.5}
            ]
        }
    }]

# --- Load audio and YAMNet timeline ---
y, sr = librosa.load(audio_file, sr=44100, mono=True)
with open(yamnet_file) as f:
    timeline = json.load(f)

# --- Compute features (must match timeline length) ---
# Match the number of frames with timeline windows (use their settings if possible)
N = len(timeline)
duration = librosa.get_duration(y=y, sr=sr)
win_sec = timeline[1]['time'] - timeline[0]['time'] if N > 1 else 1.0  # window size
frame_length = int(win_sec * sr)
hop_length = frame_length  # Non-overlapping

rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length, n_fft=frame_length)
# Pad to match timeline if needed
while len(rms) < N:
    rms = np.append(rms, rms[-1])
    cent = np.append(cent, cent[-1])
    times = np.append(times, times[-1] + win_sec)

# Normalize features
intensity = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
sharpness = (cent - cent.min()) / (cent.max() - cent.min() + 1e-6)

# --- Build hybrid haptic pattern ---
ahap_events = []
for i in range(N):
    t = timeline[i]['time']
    label = timeline[i]['label']
    label_simple = EVENT_CATEGORIES.get(label, None)
    # Masked class: no haptics!
    if label in MASK_CLASSES:
        continue
    # Special effect event
    if label_simple == "Explosion":
        ahap_events.extend(make_explosion(t))
    elif label_simple == "Gunfire":
        ahap_events.extend(make_gunfire(t))
    elif label_simple == "Music":
        ahap_events.extend(make_music(t, duration=win_sec))
    elif label_simple == "Engine":
        ahap_events.extend(make_engine(t, duration=win_sec, y=y, sr=sr))
    elif label_simple == "Whoosh":
        ahap_events.extend(make_whoosh(t))
    else:
        # Feature-based mapping (skip very low intensity)
        inten = float(intensity[i])
        sharp = float(sharpness[i])
        if inten > 0.1:
            ahap_events.append({
                "Event": {
                    "Time": round(t, 3),
                    "EventType": "HapticTransient",
                    "EventParameters": [
                        {"ParameterID": "HapticIntensity", "ParameterValue": inten},
                        {"ParameterID": "HapticSharpness", "ParameterValue": sharp}
                    ]
                }
            })

ahap = {
    "Version": 1,
    "Pattern": ahap_events
}

with open(output_ahap, "w") as f:
    json.dump(ahap, f, indent=2)

print(f"Generated {output_ahap} with {len(ahap_events)} events.")
