# audio_pipeline/pipeline.py

import os
import subprocess
from audio_pipeline.modules.treat_flac_tracks import chunk_audio
from audio_pipeline.modules.correct_json_output import correct_whisperx_outputs
from audio_pipeline.modules.convert_json_to_srt import convert_json_folder_to_srt
from audio_pipeline.modules.merge_srt_by_chunk import merge_srt_by_chunk
from audio_pipeline.modules.merge_speaker_entries import merge_speaker_entries
from audio_pipeline.modules.convert_srt_to_script import srt_to_script

WHISPERX_DOCKER_IMAGE = "atp-whisperX-ubuntu22.04-cuda:11.8.0-cudnn8"

def run_whisperx_docker(input_dir: str, output_dir: str) -> None:
    """
    Run WhisperX transcription using Docker.

    Args:
        input_dir (str): Folder containing audio chunks to transcribe.
        output_dir (str): Folder to save WhisperX output JSONs.
    """
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "docker", "run", "--gpus", "all",
        "-v", f"{os.path.abspath(input_dir)}:/input",
        "-v", f"{os.path.abspath(output_dir)}:/output",
        WHISPERX_DOCKER_IMAGE,
        "--diarize", "False",
        "--output_dir", "/output",
        "--language", "en",
        "--input_dir", "/input",
        "--batch_size", "4",
        "--compute_type", "float16"
    ]

    print(f"Running WhisperX Docker:\n{' '.join(command)}")
    subprocess.run(command, check=True)
    print("WhisperX transcription complete.")


def run_pipeline(
    base_path: str = ".",
    input_audio_folder: str = "tracks",
    whisperx_output_folder: str = "wx_output",
    corrected_json_folder: str = "json_files",
    srt_folder: str = "srt_files",
    final_folder: str = "final_outputs",
    merged_audio_filename: str = "merged_audio.flac",
    cut_points_filename: str = "cut_points.txt",
    merged_transcript_filename: str = "merged_transcript.srt",
    cleaned_transcript_filename: str = "cleaned_transcript.srt",
    final_script_filename: str = "final_script.txt",
    use_low_ram: bool = False,
    run_whisperx: bool = True
) -> None:

    """
    Run the full audio transcription pipeline.

    Args:
        base_path (str): Base path where all folders will be created.
        input_audio_folder (str): Folder containing raw audio tracks.
        whisperx_output_folder (str): Folder containing WhisperX outputs.
        corrected_json_folder (str): Folder for corrected JSONs.
        srt_folder (str): Folder for generated SRTs.
        final_folder (str): Folder for final outputs.
        merged_audio_filename (str): Filename for merged audio file.
        cut_points_filename (str): Filename for cut points text file.
        merged_transcript_filename (str): Filename for merged SRT file.
        cleaned_transcript_filename (str): Filename for cleaned SRT file.
        final_script_filename (str): Filename for final readable script.
        use_low_ram (bool): Enable low RAM processing for audio chunking.
        run_whisperx (bool): 
            If True, automatically run WhisperX transcription inside a Docker container.
            If False, prompt the user to manually transcribe audio chunks before continuing.
    """
    base_path = os.path.abspath(base_path)

    input_audio_folder = os.path.join(base_path, input_audio_folder)
    whisperx_output_folder = os.path.join(base_path, whisperx_output_folder)
    corrected_json_folder = os.path.join(base_path, corrected_json_folder)
    srt_folder = os.path.join(base_path, srt_folder)
    final_folder = os.path.join(base_path, final_folder)

    os.makedirs(final_folder, exist_ok=True)

    print("\n=== Step 1: Chunking audio tracks ===")
    chunk_audio(
        input_folder=input_audio_folder,
        output_dir=os.path.join(final_folder, "sepchannel"),
        merged_file=os.path.join(final_folder, merged_audio_filename),
        cut_points_file=os.path.join(final_folder, cut_points_filename),
        use_low_ram=use_low_ram
    )

    print("\n=== Step 2: WhisperX transcription ===")
    if run_whisperx:
        run_whisperx_docker(
            input_dir=os.path.join(final_folder, "sepchannel"),
            output_dir=whisperx_output_folder
        )
    else:
        print("Please transcribe the generated 'sepchannel/' files manually (e.g., using WhisperX).")
        input("Press Enter once WhisperX outputs are available in 'wx_output/'...")

    print("\n=== Step 3: Correcting WhisperX output ===")
    correct_whisperx_outputs(
        input_dir=whisperx_output_folder,
        output_dir=corrected_json_folder
    )

    print("\n=== Step 4: Converting corrected JSON to SRT ===")
    convert_json_folder_to_srt(
        input_dir=corrected_json_folder,
        output_dir=srt_folder
    )

    print("\n=== Step 5: Merging SRT files by chunk ===")
    merge_srt_by_chunk(
        srt_folder=srt_folder,
        output_dir=final_folder,
        output_filename=merged_transcript_filename
    )

    print("\n=== Step 6: Merging successive speaker entries ===")
    merge_speaker_entries(
        input_srt_file=os.path.join(final_folder, merged_transcript_filename),
        output_srt_file=os.path.join(final_folder, cleaned_transcript_filename)
    )

    print("\n=== Step 7: Creating final readable script ===")
    srt_to_script(
        input_srt_path=os.path.join(final_folder, cleaned_transcript_filename),
        output_script_path=os.path.join(final_folder, final_script_filename)
    )

    print("\n✅ Pipeline finished successfully! Outputs available in:", final_folder)
