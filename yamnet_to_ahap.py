import json

# Load timeline
with open("yamnet_timeline.json") as f:
    timeline = json.load(f)

# Map YAMNet label to AHAP category
LABEL_TO_HAPTIC = {
    "Explosion": "Explosion",
    "Boom": "Explosion",
    "Eruption": "Explosion",
    "Gunshot, gunfire": "Gunfire",
    "Artillery fire": "Gunfire",
    "Music": "Music",
    "Vehicle": "Rumble",
    "Aircraft": "Rumble",
    "Boat, Water vehicle": "Rumble",
    "Whoosh, swoosh, swish": "Whoosh",
    "Wind": "Whoosh",
    # add more as needed
}

def make_rumble(start_time, duration=1.0):
    # Clone the "Rumble" example pattern, shifting times by start_time
    base = []
    for i in range(0, int(duration * 25)):  # Rumble events every 0.04s
        t = round(start_time + i * 0.04, 3)
        base.append({
            "Event": {
                "Time": t,
                "EventType": "HapticTransient",
                "EventParameters": [
                    {"ParameterID": "HapticIntensity", "ParameterValue": 1.0},
                    {"ParameterID": "HapticSharpness", "ParameterValue": 0.1}
                ]
            }
        })
    # Add intensity curve for realism
    base.insert(0, {
        "ParameterCurve": {
            "ParameterID": "HapticIntensityControl",
            "Time": start_time,
            "ParameterCurveControlPoints": [
                { "Time": start_time, "ParameterValue": 0.3 },
                { "Time": start_time + 0.1, "ParameterValue": 1.0 },
                { "Time": start_time + 0.8, "ParameterValue": 1.0 },
                { "Time": start_time + duration, "ParameterValue": 0.0 }
            ]
        }
    })
    return base

def make_inflate(start_time, duration=1.7):
    # Clone the "Inflate" pattern, shifting times by start_time
    return [
        {
            "Event": {
                "Time": start_time,
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
                "Time": start_time,
                "ParameterCurveControlPoints": [
                    { "Time": start_time, "ParameterValue": 0.0 },
                    { "Time": start_time + 1.1, "ParameterValue": 0.5 },
                    { "Time": start_time + duration, "ParameterValue": 0.0 }
                ]
            }
        },
        {
            "ParameterCurve": {
                "ParameterID": "HapticSharpnessControl",
                "Time": start_time,
                "ParameterCurveControlPoints": [
                    { "Time": start_time, "ParameterValue": -0.8 },
                    { "Time": start_time + duration, "ParameterValue": 0.8 }
                ]
            }
        }
    ]

# AHAP event patterns for each category
def category_to_ahap(label, time):
    if label == "Explosion":
        return [{
            "Event": {
                "Time": time,
                "EventType": "HapticTransient",
                "EventParameters": [
                    {"ParameterID": "HapticIntensity", "ParameterValue": 1.0},
                    {"ParameterID": "HapticSharpness", "ParameterValue": 1.0}
                ]
            }
        }]
    elif label == "Gunfire":
        return [{
            "Event": {
                "Time": time,
                "EventType": "HapticTransient",
                "EventParameters": [
                    {"ParameterID": "HapticIntensity", "ParameterValue": 0.8},
                    {"ParameterID": "HapticSharpness", "ParameterValue": 1.0}
                ]
            }
        }]
    elif label == "Rumble":
        return make_rumble(time, duration=1.0)
    elif label == "Music":
        return make_inflate(time, duration=1.5)
    elif label == "Whoosh":
        return [{
            "Event": {
                "Time": time,
                "EventType": "HapticTransient",
                "EventParameters": [
                    {"ParameterID": "HapticIntensity", "ParameterValue": 0.7},
                    {"ParameterID": "HapticSharpness", "ParameterValue": 0.5}
                ]
            }
        }]
    else:
        return []

ahap_pattern = []
for event in timeline:
    label = LABEL_TO_HAPTIC.get(event["label"])
    if label:
        ahap_pattern.extend(category_to_ahap(label, event["time"]))

ahap = {
    "Version": 1,
    "Pattern": ahap_pattern
}

with open("pattern.ahap", "w") as f:
    json.dump(ahap, f, indent=2)

print(f"Generated pattern.ahap with {len(ahap_pattern)} events/patterns.")
