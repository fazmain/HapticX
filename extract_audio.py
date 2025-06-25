# Code to extract audio from a video file

import ffmpeg

video_file = "input.mp4"
audio_file = "audio.wav"

# Extract audio using ffmpeg
(
    ffmpeg
    .input(video_file)
    .output(audio_file, acodec='pcm_s16le', ac=1, ar='44100')  # mono, 44.1kHz
    .overwrite_output()
    .run()
)

print(f"Audio extracted and saved to {audio_file}")
