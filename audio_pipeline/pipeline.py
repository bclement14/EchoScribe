# audio_pipeline/pipeline.py

import os
import subprocess
from audio_pipeline.modules.treat_flac_tracks import chunk_audio
from audio_pipeline.modules.correct_json_output import correct_whisperx_outputs
from audio_pipeline.modules.convert_json_to_srt import convert_json_folder_to_srt
from audio_pipeline.modules.merge_srt_by_chunk import merge_srt_by_chunk
from audio_pipeline.modules.merge_speaker_entries import merge_speaker_entries
from audio_pipeline.modules.convert_srt_to_script import srt_to_script

WHISPERX_DOCKER_IMAGE = "echoscribe-whisperx"

import os
import subprocess
from typing import Optional

def run_whisperx_docker(
    session_path: str,
    chunks_folder: str,
    diarize: bool = False,
    huggingface_token: Optional[str] = None
) -> None:
    """
    Run WhisperX transcription using Docker.

    Args:
        session_path (str): Path to the working session folder (where tracks/, chunked_tracks/, etc. exist).
        diarize (bool): Enable or disable diarization (speaker recognition).
        huggingface_token (Optional[str]): Hugging Face token, required if diarization is enabled.
    """
    abs_session_path = os.path.abspath(session_path)

    command = [
        "docker", "run", "--gpus", "all", "-it",
        "-v", f"{abs_session_path}:/app",
        "--entrypoint", "whisperx",
        "echoscribe-whisperx",
        "--diarize", str(diarize),
        "--model", "large-v3",
        "--language", "fr",
        "--output_dir", "wx_output",
        "--output_format", "json",
        f"{chunks_folder}/*.flac"
    ]

    if diarize:
        if not huggingface_token:
            raise ValueError("Hugging Face token must be provided if diarization is enabled.")
        command.insert(-1, "--hf_token")
        command.insert(-1, huggingface_token)

    print(f"Running WhisperX Docker:\n{' '.join(command)}")
    subprocess.run(command, check=True)
    print("WhisperX transcription complete.")



def run_pipeline(
    base_path: str = ".",
    input_audio_folderName: str = "tracks",
    chunks_audio_folderName: str = "chunked_tracks",
    whisperx_output_folderName: str = "wx_output",
    corrected_json_folderName: str = "json_files",
    srt_folderName: str = "srt_files",
    final_folderName: str = "final_outputs",
    merged_audio_filename: str = "merged_audio.flac",
    cut_points_filename: str = "cut_points.txt",
    merged_transcript_filename: str = "merged_transcript.srt",
    cleaned_transcript_filename: str = "cleaned_transcript.srt",
    final_script_filename: str = "final_script.txt",
    use_low_ram: bool = False,
    run_whisperx: bool = True,
    diarize: bool = False,
    huggingface_token: Optional[str] = None
) -> None:
    """
    Run the full audio transcription pipeline.

    Args:
        base_path (str): Base path where all folders will be created.
        input_audio_folderName (str): Folder containing raw audio tracks.
        chunks_audio_folderName: (str): Folder containing tracks chunked.
        whisperx_output_folderName (str): Folder containing WhisperX outputs.
        corrected_json_folderName (str): Folder for corrected JSONs.
        srt_folderName (str): Folder for generated timestamped transcripts.
        final_folderName (str): Folder for final outputs.
        merged_audio_filename (str): Filename for merged audio file (.flac).
        cut_points_filename (str): Filename for silence detection cut points (.txt).
        merged_transcript_filename (str): Filename for merged transcript (.srt).
        cleaned_transcript_filename (str): Filename for cleaned merged transcript (.srt).
        final_script_filename (str): Filename for final readable script (.txt).
        use_low_ram (bool): Enable low RAM processing for audio chunking.
        run_whisperx (bool): 
            If True, automatically run WhisperX transcription inside Docker.
            If False, prompt user to transcribe manually before continuing.
        diarize (bool): 
            If True, enable speaker diarization (requires Hugging Face token).
        huggingface_token (Optional[str]): 
            Hugging Face token to access speaker models for diarization. 
            If not provided and diarization is enabled, the pipeline will attempt to load it from the HF_TOKEN environment variable.
    """

    base_path = os.path.abspath(base_path)
    input_audio_folder = os.path.join(base_path, input_audio_folderName)
    chunked_audio_folder = os.path.join(base_path, chunks_audio_folderName)
    whisperx_output_folder = os.path.join(base_path, whisperx_output_folderName)
    corrected_json_folder = os.path.join(base_path, corrected_json_folderName)
    srt_folder = os.path.join(base_path, srt_folderName)
    final_folder = os.path.join(base_path, final_folderName)

    os.makedirs(final_folder, exist_ok=True)

    print("\n=== Step 1: Chunking audio tracks ===")
    chunk_audio(
        input_folder=input_audio_folder,
        output_dir=chunked_audio_folder,
        merged_file=os.path.join(final_folder, merged_audio_filename),
        cut_points_file=os.path.join(final_folder, cut_points_filename),
        use_low_ram=use_low_ram
    )

    print("\n=== Step 2: WhisperX transcription ===")
    if run_whisperx:
        if diarize and not huggingface_token:
            huggingface_token = os.environ.get("HF_TOKEN")
            if huggingface_token is None:
                raise ValueError(
                    "Diarization is enabled but no Hugging Face token was provided "
                    "and 'HF_TOKEN' environment variable is not set."
                )

        run_whisperx_docker(
            session_path=os.path.join(base_path),
            chunks_folder=chunks_audio_folderName,
            diarize=diarize,
            huggingface_token=huggingface_token
        )
    else:
        print(f"Please transcribe the generated '{chunks_audio_folderName}/' files manually (e.g., using WhisperX).")
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
