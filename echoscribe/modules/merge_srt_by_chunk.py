# echoscribe/modules/merge_srt_by_chunk.py

import os
import glob
import srt
from datetime import timedelta
from tqdm import tqdm
from typing import List

def load_srt_file(file_path: str, speaker_name: str, time_offset: timedelta) -> List[srt.Subtitle]:
    """
    Load an SRT file, adjust timestamps based on time_offset, and tag with speaker name.
    
    Args:
        file_path (str): Path to the SRT file.
        speaker_name (str): Name of the speaker to tag each subtitle.
        time_offset (timedelta): Offset to apply to each subtitle's timestamps.

    Returns:
        List[srt.Subtitle]: List of adjusted subtitles.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        subtitles = list(srt.parse(file.read()))

    for subtitle in subtitles:
        subtitle.start += time_offset
        subtitle.end += time_offset
        subtitle.content = f"[{speaker_name}] {subtitle.content}"

    return subtitles


def merge_srt_by_chunk(srt_folder: str, output_dir: str, output_filename: str) -> None:
    """
    Merge all SRT files by chunk while adjusting timestamps.

    Args:
        srt_folder (str): Folder containing individual SRT files.
        output_dir (str): Folder where the merged SRT will be saved.
        output_filename (str): Name of the final merged SRT file.
    """
    print(f"Merging SRT files from '{srt_folder}' into '{output_dir}/{output_filename}'...")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_filename)
    all_subtitles = []
    time_offset = timedelta(0)  # Start at 00:00:00

    # Organize SRT files by chunk number
    chunk_files = {}  # {chunk_number: {speaker_name: file_path}}
    for file_path in glob.glob(os.path.join(srt_folder, "*.srt")):
        filename = os.path.basename(file_path)
        parts = filename.rsplit("-", 1)  # Format: speaker-chunk.srt
        if len(parts) == 2 and parts[1].replace(".srt", "").isdigit():
            speaker_name = parts[0]
            chunk_number = int(parts[1].replace(".srt", ""))
            if chunk_number not in chunk_files:
                chunk_files[chunk_number] = {}
            chunk_files[chunk_number][speaker_name] = file_path

    print(f"Found {len(chunk_files)} chunks to process.")

    for chunk_number in tqdm(sorted(chunk_files.keys()), desc="Merging chunks"):
        current_chunk_subs = []

        for speaker_name, file_path in chunk_files[chunk_number].items():
            subtitles = load_srt_file(file_path, speaker_name, time_offset)
            current_chunk_subs.extend(subtitles)

        # Sort subtitles inside the current chunk
        current_chunk_subs.sort(key=lambda x: x.start)

        # Add current chunk's subtitles to the global list
        all_subtitles.extend(current_chunk_subs)

        # Update time offset based on last subtitle end + small buffer
        if current_chunk_subs:
            time_offset = all_subtitles[-1].end + timedelta(seconds=1)

    # Save the merged subtitles
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(srt.compose(all_subtitles))

    print(f"Merged subtitles saved to '{output_path}'.")