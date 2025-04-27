from pathlib import Path
from typing import List, Tuple
from pydub import AudioSegment, silence
import os
from tqdm import tqdm

def get_audio_files_from_folder(folder: str, extension: str = "flac") -> List[str]:
    """Retrieve all audio files with the specified extension from the folder."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    return [str(file) for file in folder_path.glob(f"*.{extension}")]


def merge_tracks(
    speaker_files: List[str], 
    output_file: str, 
    use_low_ram: bool = False, 
    chunk_size: int = 60000
) -> str:
    """Merge multiple speaker audio files into a single track."""
    if use_low_ram:
        return merge_tracks_incremental(speaker_files, output_file, chunk_size)
    else:
        print("Merging tracks into one (high RAM mode)...")
        tracks = [AudioSegment.from_file(file, format="flac") for file in speaker_files]
        merged = tracks[0]
        for track in tracks[1:]:
            merged = merged.overlay(track)
        merged.export(output_file, format="flac")
        print(f"Merged track saved to {output_file}")
        return output_file


def merge_tracks_incremental(
    speaker_files: List[str], 
    output_file: str, 
    chunk_size: int = 60000
) -> str:
    """Incrementally merge multiple audio files to reduce RAM usage."""
    print("Merging tracks into one (low RAM mode)...")
    tracks = [AudioSegment.from_file(file, format="flac") for file in tqdm(speaker_files, desc="Loading tracks")]
    merged = AudioSegment.silent(duration=0)

    max_length = max(len(track) for track in tracks)
    for i in tqdm(range(0, max_length, chunk_size), desc="Merging chunks"):
        chunk = sum(track[i:i + chunk_size] for track in tracks if i < len(track))
        merged += chunk

    merged.export(output_file, format="flac")
    print(f"Merged track saved to {output_file}")
    return output_file


def detect_silence_in_audio(
    audio_file: str, 
    silence_threshold: int = -40, 
    min_silence_duration: int = 1000,
    use_low_ram: bool = False,
    chunk_size: int = 60000
) -> List[Tuple[float, float]]:
    """Detect silence in an audio file."""
    if use_low_ram:
        return detect_silence_incremental(audio_file, silence_threshold, min_silence_duration, chunk_size)
    else:
        print("Detecting silences (high RAM mode)...")
        audio = AudioSegment.from_file(audio_file, format="flac")
        silent_ranges = silence.detect_silence(
            audio,
            min_silence_len=min_silence_duration,
            silence_thresh=silence_threshold
        )
        print(f"Detected {len(silent_ranges)} silent ranges.")
        return [(start / 1000, end / 1000) for start, end in silent_ranges]


def detect_silence_incremental(
    audio_file: str,
    silence_threshold: int = -40,
    min_silence_duration: int = 1000,
    chunk_size: int = 60000
) -> List[Tuple[float, float]]:
    """Detect silence incrementally to reduce RAM usage."""
    print("Detecting silences (low RAM mode)...")
    audio = AudioSegment.from_file(audio_file, format="flac")
    silent_ranges = []

    for i in tqdm(range(0, len(audio), chunk_size), desc="Analyzing chunks"):
        chunk = audio[i:i + chunk_size]
        chunk_silences = silence.detect_silence(
            chunk,
            min_silence_len=min_silence_duration,
            silence_thresh=silence_threshold
        )
        silent_ranges.extend([(start + i) / 1000, (end + i) / 1000] for start, end in chunk_silences)

    print(f"Detected {len(silent_ranges)} silent ranges.")
    return silent_ranges


def determine_cut_points(
    silence_timestamps: List[Tuple[float, float]],
    min_duration: int = 9 * 60,
    max_duration: int = 15 * 60
) -> List[float]:
    """Select best cut points based on detected silences."""
    print("Determining cut points...")
    cut_points = [0]
    last_cut = 0

    for start, end in silence_timestamps:
        if end - last_cut >= min_duration:
            if end - last_cut <= max_duration:
                cut_points.append(end)
                last_cut = end
            else:
                closest_valid_cut = min(end, last_cut + max_duration)
                cut_points.append(closest_valid_cut)
                last_cut = closest_valid_cut

    print(f"Determined {len(cut_points)-1} cut segments.")
    return cut_points


def write_cut_points(cut_points: List[float], output_file: str) -> None:
    """Write cut points to a text file."""
    print(f"Writing cut points to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for i, timestamp in enumerate(cut_points):
            f.write(f"Chunk {i}: {timestamp:.2f} seconds\n")
    print("Cut points saved.")


def split_tracks(
    speaker_files: List[str],
    cut_points: List[float],
    output_dir: str,
    use_low_ram: bool = False
) -> None:
    """Split speaker tracks at the given timestamps and save them."""
    print(f"Splitting tracks into {len(cut_points) - 1} segments...")
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(speaker_files, desc="Splitting files"):
        filename = Path(file).stem
        audio = AudioSegment.from_file(file, format="flac")

        for i in range(len(cut_points) - 1):
            start_ms = int(cut_points[i] * 1000)
            end_ms = int(cut_points[i + 1] * 1000)
            segment = audio[start_ms:end_ms]

            indice = f"{i+1:02d}"
            output_filename = f"{filename}-{indice}.flac"
            segment.export(Path(output_dir) / output_filename, format="flac")


def chunk_audio(
    input_folder: str,
    output_dir: str,
    merged_file: str,
    cut_points_file: str,
    use_low_ram: bool = False
) -> None:
    """Pipeline to chunk audio files."""
    print("Starting audio chunking pipeline...")

    speaker_files = get_audio_files_from_folder(input_folder)

    if not speaker_files:
        print(f"No FLAC files found in {input_folder}.")
        return

    merge_tracks(speaker_files, merged_file, use_low_ram=use_low_ram)
    silence_timestamps = detect_silence_in_audio(merged_file, use_low_ram=use_low_ram)
    cut_points = determine_cut_points(silence_timestamps)
    write_cut_points(cut_points, cut_points_file)
    split_tracks(speaker_files, cut_points, output_dir, use_low_ram=use_low_ram)

    print(f"Audio chunking pipeline completed. Output at {output_dir}")
