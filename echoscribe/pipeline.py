# echoscribe/pipeline.py

import os
import subprocess
from typing import Optional, Union, List, Tuple
from .modules import chunk_audio, correct_whisperx_outputs, convert_json_folder_to_srt, merge_srt_by_chunk, merge_speaker_entries, srt_to_script
import glob

WHISPERX_DOCKER_IMAGE = "echoscribe-whisperx"



def should_run_step(step_number: int, steps_to_run: Union[slice, List[int], Tuple[int]]) -> bool:
    """
    Determine whether a given pipeline step should run based on the selected steps.

    Args:
        step_number (int): The step number (starting from 1) to check.
        steps_to_run (Union[slice, list, tuple]): Slice or list/tuple of steps to run.

    Returns:
        bool: True if the step should be executed, False otherwise.
    """
    if isinstance(steps_to_run, slice):
        return step_number in range(*steps_to_run.indices(100))
    if isinstance(steps_to_run, (list, tuple)):
        return step_number in steps_to_run
    return False

def run_whisperx_docker(
    session_path: str,
    chunks_folder: str,
    diarize: bool = False,
    huggingface_token: Optional[str] = None
) -> None:
    """
    Run WhisperX transcription using Docker.
    """
    abs_session_path = os.path.abspath(session_path)
    
    # Change this line - don't use wildcards in the Docker command
    command = [
        "docker", "run", "--gpus", "all", "-it",
        "-v", f"{abs_session_path}:/app",
        "--entrypoint", "bash",
        f"{WHISPERX_DOCKER_IMAGE}",
        "-c", f"cd /app && whisperx /app/{chunks_folder}/*.flac --model large-v3 --language fr --output_dir wx_output --output_format json"
    ]
    
    if diarize:
        # Insert diarization options into the bash command rather than the argument list
        diarize_options = "--diarize"
        if not huggingface_token:
            raise ValueError("Hugging Face token must be provided if diarization is enabled.")
        diarize_options += f" --hf_token {huggingface_token}"
        # Update the bash command to include diarization
        command[-1] = command[-1].replace("--output_format json", f"--output_format json {diarize_options}")
    
    print(f"Running WhisperX Docker:\n{' '.join(command)}")
    # Run the command with shell=True
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
    use_low_ram: bool = True,
    run_whisperx: bool = True,
    diarize: bool = False,
    huggingface_token: Optional[str] = None,
    steps_to_run: Union[slice, List[int], Tuple[int]] = slice(0, None)
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
        run_whisperx (bool): Run WhisperX automatically if True, else manual transcription.
        diarize (bool): Enable speaker diarization (requires Hugging Face token).
        huggingface_token (Optional[str]): Hugging Face token for diarization, auto-loaded from HF_TOKEN env var if not provided.
        steps_to_run (slice or list): Select which pipeline steps to run.
    """
    base_path = os.path.abspath(base_path)

    input_audio_folder = os.path.join(base_path, input_audio_folderName)
    chunked_audio_folder = os.path.join(base_path, chunks_audio_folderName)
    whisperx_output_folder = os.path.join(base_path, whisperx_output_folderName)
    corrected_json_folder = os.path.join(base_path, corrected_json_folderName)
    srt_folder = os.path.join(base_path, srt_folderName)
    final_folder = os.path.join(base_path, final_folderName)

    os.makedirs(final_folder, exist_ok=True)

    steps_executed = []

    if should_run_step(1, steps_to_run):
        steps_executed.append(1)
        print("\n=== Step 1: Chunking audio tracks ===")
        chunk_audio(
            input_folder=input_audio_folder,
            output_dir=chunked_audio_folder,
            merged_file=os.path.join(final_folder, merged_audio_filename),
            cut_points_file=os.path.join(final_folder, cut_points_filename),
            use_low_ram=use_low_ram
        )

    if should_run_step(2, steps_to_run):
        steps_executed.append(2)
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

    if should_run_step(3, steps_to_run):
        steps_executed.append(3)
        print("\n=== Step 3: Correcting WhisperX output ===")
        correct_whisperx_outputs(
            input_dir=whisperx_output_folder,
            output_dir=corrected_json_folder
        )
    

    if should_run_step(4, steps_to_run):
        steps_executed.append(4)
        print("\n=== Step 4: Converting corrected JSON to SRT ===")
        convert_json_folder_to_srt(
            input_dir=corrected_json_folder,
            output_dir=srt_folder
        )

    if should_run_step(5, steps_to_run):
        steps_executed.append(5)
        print("\n=== Step 5: Merging SRT files by chunk ===")
        merge_srt_by_chunk(
            srt_folder=srt_folder,
            output_dir=final_folder,
            output_filename=merged_transcript_filename
        )

    if should_run_step(6, steps_to_run):
        steps_executed.append(6)
        print("\n=== Step 6: Merging successive speaker entries ===")
        merge_speaker_entries(
            input_srt_file=os.path.join(final_folder, merged_transcript_filename),
            output_srt_file=os.path.join(final_folder, cleaned_transcript_filename)
        )

    if should_run_step(7, steps_to_run):
        steps_executed.append(7)
        print("\n=== Step 7: Creating final readable script ===")
        srt_to_script(
            input_srt_path=os.path.join(final_folder, cleaned_transcript_filename),
            output_script_path=os.path.join(final_folder, final_script_filename)
        )

    if steps_executed == list(range(1, 8)):
        print("\n✅ Full pipeline finished successfully! Outputs available in:", final_folder)
    else:
        print(f"\n✅ Selected pipeline steps {steps_executed} finished successfully! Outputs available in:", final_folder)

