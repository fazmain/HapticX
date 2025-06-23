import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import csv
import json

labels_path = 'yamnet_class_map.csv'
with open(labels_path, 'r') as f:
    reader = csv.DictReader(f)
    class_map = [row['display_name'] for row in reader]

model = hub.load('https://tfhub.dev/google/yamnet/1')
audio_file = 'audio.wav'
wav, sr = librosa.load(audio_file, sr=16000, mono=True)

window_duration = 1.0  # seconds
hop_duration = 0.5     # seconds
window_length = int(window_duration * sr)
hop_length = int(hop_duration * sr)

timeline = []

for start in range(0, len(wav) - window_length, hop_length):
    segment = wav[start:start + window_length]
    scores, _, _ = model(segment)
    mean_scores = np.mean(scores.numpy(), axis=0)
    top_class = mean_scores.argmax()
    top_label = class_map[top_class]
    confidence = float(mean_scores[top_class])
    event_time = round(start / sr, 2)
    timeline.append({
        "time": event_time,
        "label": top_label,
        "confidence": confidence
    })

with open("yamnet_timeline.json", "w") as f:
    json.dump(timeline, f, indent=2)

print("Saved yamnet_timeline.json")

