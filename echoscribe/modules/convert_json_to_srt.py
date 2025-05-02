# echoscribe/modules/convert_json_to_srt.py

import os
import json
from tqdm import tqdm
from typing import List, Dict

def format_srt_time(seconds: float) -> str:
    """Convert seconds (float) to SRT time format "HH:MM:SS,ms"."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    
    # Handle rare case where milliseconds round to 1000
    if milliseconds == 1000:
        milliseconds = 0
        sec += 1
        if sec == 60:
            sec = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1

    return f"{hours:02d}:{minutes:02d}:{sec:02d},{milliseconds:03d}"


def json_to_srt(json_path: str, srt_path: str) -> None:
    """Convert a single WhisperX JSON file to an SRT file."""
    with open(json_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    segments = data.get("segments", [])
    srt_blocks = []
    index = 1

    for segment in segments:
        start_time = format_srt_time(segment["start"])
        end_time = format_srt_time(segment["end"])
        text = segment.get("text", "").strip()
        
        if not text and "words" in segment:
            text = " ".join(w.get("word", "") for w in segment["words"]).strip()
        
        block = f"{index}\n{start_time} --> {end_time}\n{text}\n"
        srt_blocks.append(block)
        index += 1

    with open(srt_path, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(srt_blocks))


def convert_json_folder_to_srt(input_dir: str, output_dir: str) -> None:
    """
    Convert all corrected JSON files in a folder to SRT subtitle files.
    
    Args:
        input_dir (str): Path to corrected JSON folder.
        output_dir (str): Path to save generated SRT files.
    """
    print(f"Converting corrected JSON files from '{input_dir}' to SRT format in '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    for filename in tqdm(files, desc="Converting JSON to SRT"):
        json_path = os.path.join(input_dir, filename)
        srt_filename = os.path.splitext(filename)[0] + ".srt"
        srt_path = os.path.join(output_dir, srt_filename)
        json_to_srt(json_path, srt_path)

    print("Conversion to SRT complete.")
