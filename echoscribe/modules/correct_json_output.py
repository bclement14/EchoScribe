# audio_pipeline/modules/correct_json_output.py

import os
import json
from tqdm import tqdm
from typing import List, Dict

# Constants
MAX_WORD_DURATION = 2.5  # seconds
MAX_GAP_DURATION = 5     # seconds
CORRECTION_DURATION = 1  # seconds
PUNCTUATION = {".", ",", "?", "!", ":", ";"}

def process_segment(segment: Dict) -> List[Dict]:
    """
    Process a single segment and return a list of corrected segments.
    Splits the segment at each error (duration or gap error), unless the next word is punctuation.
    """
    words = segment["words"]
    corrected_segments = []
    current_words = []
    current_start = segment["start"]

    i = 0
    while i < len(words):
        word = words[i]
        duration = word["end"] - word["start"]
        error_detected = False

        # Check duration error
        if duration > MAX_WORD_DURATION:
            if i == 0:
                word["start"] = word["end"] - CORRECTION_DURATION
            elif i == len(words) - 1:
                word["end"] = word["start"] + CORRECTION_DURATION
            else:
                word["end"] = word["start"] + CORRECTION_DURATION
                error_detected = True

        # Check gap error
        if not error_detected and i < len(words) - 1:
            gap = words[i+1]["start"] - word["end"]
            if gap > MAX_GAP_DURATION:
                word["end"] = word["start"] + CORRECTION_DURATION
                error_detected = True

        current_words.append(word)

        if error_detected and i != len(words) - 1:
            next_word_text = words[i+1]["word"].strip()
            if next_word_text in PUNCTUATION:
                words[i+1]["start"] = word["end"]
                punct_duration = words[i+1]["end"] - words[i+1]["start"]
                if punct_duration > MAX_WORD_DURATION:
                    words[i+1]["end"] = words[i+1]["start"] + CORRECTION_DURATION
                error_detected = False
            else:
                # Split segment
                new_segment = {
                    "start": current_start,
                    "end": word["end"],
                    "words": current_words.copy(),
                    "text": " ".join(w["word"] for w in current_words)
                }
                corrected_segments.append(new_segment)
                current_words = []
                current_start = words[i+1]["start"]

        i += 1

    if current_words:
        new_segment = {
            "start": current_start,
            "end": current_words[-1]["end"],
            "words": current_words.copy(),
            "text": " ".join(w["word"] for w in current_words)
        }
        corrected_segments.append(new_segment)

    return corrected_segments


def process_file(input_path: str, output_path: str) -> None:
    """Process a single JSON file."""
    with open(input_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    new_segments = []
    for seg in data.get("segments", []):
        fixed_segments = process_segment(seg)
        new_segments.extend(fixed_segments)

    data["segments"] = new_segments

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)


def correct_whisperx_outputs(input_dir: str, output_dir: str) -> None:
    """
    Correct all WhisperX JSON files in a folder.
    
    Args:
        input_dir (str): Path to folder with raw WhisperX output JSONs.
        output_dir (str): Path to save corrected JSONs.
    """
    print(f"Correcting WhisperX outputs from '{input_dir}' to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    for filename in tqdm(files, desc="Correcting JSON files"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        process_file(input_path, output_path)

    print("Correction complete.")
