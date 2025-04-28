# audio_pipeline/modules/merge_speaker_entries.py

import srt
from datetime import timedelta
from typing import List
import os

def merge_speaker_entries(input_srt_file: str, output_srt_file: str) -> None:
    """
    Merge consecutive subtitles from the same speaker into a single entry.

    Args:
        input_srt_file (str): Path to the merged SRT file.
        output_srt_file (str): Path to save the cleaned SRT file.
    """
    print(f"Merging successive speaker entries in '{input_srt_file}'...")

    if not os.path.isfile(input_srt_file):
        raise FileNotFoundError(f"Input SRT file not found: {input_srt_file}")

    with open(input_srt_file, 'r', encoding='utf-8') as file:
        subtitles = list(srt.parse(file.read()))

    if not subtitles:
        print("No subtitles found in the file. Exiting.")
        return

    merged_subtitles = []
    current_entry = subtitles[0]

    for subtitle in subtitles[1:]:
        current_speaker = current_entry.content.split("]")[0] + "]"
        next_speaker = subtitle.content.split("]")[0] + "]"

        if current_speaker == next_speaker:
            # Merge the text and extend the end time
            current_entry.end = subtitle.end
            current_entry.content += " " + subtitle.content[len(current_speaker):].strip()
        else:
            merged_subtitles.append(current_entry)
            current_entry = subtitle

    # Append the last processed entry
    merged_subtitles.append(current_entry)

    # Save the cleaned SRT file
    os.makedirs(os.path.dirname(output_srt_file), exist_ok=True)
    with open(output_srt_file, 'w', encoding='utf-8') as file:
        file.write(srt.compose(merged_subtitles))

    print(f"Merged speaker entries saved to '{output_srt_file}'.")
