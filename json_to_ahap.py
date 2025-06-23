# V0 Prototype for converting simple JSON to simple AHAP with single click haptic events.

import json

def make_ahap(haptic_events, output_file="pattern.ahap"):
    pattern = {
        "Pattern": [
            {
                "Event": {
                    "Time": round(event["time"], 3),
                    "EventType": "HapticTransient",
                    "EventDuration": 0.1,
                    "EventParameters": [
                        {"ParameterID": "HapticIntensity", "ParameterValue": min(1.0, max(0.1, event["intensity"] / 0.3))},
                        {"ParameterID": "HapticSharpness", "ParameterValue": 1.0}
                    ]
                }
            } for event in haptic_events
        ]
    }
    ahap = {
        "Version": 1,
        "Pattern": pattern["Pattern"]
    }
    with open(output_file, "w") as f:
        json.dump(ahap, f, indent=2)
    print(f"Saved .ahap file as {output_file}")

# Usage:
with open("haptic_events.json") as f:
    haptic_events = json.load(f)
make_ahap(haptic_events)
