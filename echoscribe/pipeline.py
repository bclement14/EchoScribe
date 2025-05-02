# echoscribe/pipeline.py

import os
import subprocess
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field, replace # Import replace for updating dataclass
from typing import Optional, Union, List, Tuple, Sequence, Dict, Any

# Import revised modules (assuming structure and naming)
from .modules.treat_flac_tracks import chunk_audio
from .modules.whisperx_corrector_core import correct_whisperx_outputs, WhisperXCorrectorConfig # Import config too if needed later
from .modules.convert_json_to_srt import convert_json_folder_to_srt
from .modules.merge_srt_by_chunk import merge_srt_by_chunk
from .modules.merge_speaker_entries import merge_speaker_entries
from .modules.convert_srt_to_script import srt_to_script

# --- Configuration Dataclass (Still useful internally) ---

@dataclass(frozen=True) # Keep it immutable internally
class PipelineConfig:
    """Internal configuration settings for the audio processing pipeline."""
    base_path: Path = field(default_factory=Path.cwd) # Default to current dir Path object
    input_audio_folderName: str = "tracks"
    chunks_audio_folderName: str = "chunked_tracks"
    whisperx_output_folderName: str = "wx_output"
    corrected_json_folderName: str = "json_files"
    srt_folderName: str = "srt_files"
    final_folderName: str = "final_outputs"
    merged_audio_filename: str = "merged_audio.flac"
    cut_points_filename: str = "cut_points.txt"
    merged_transcript_filename: str = "merged_transcript.srt"
    cleaned_transcript_filename: str = "cleaned_transcript.srt"
    final_script_filename: str = "final_script.txt"

    use_low_ram: bool = True
    run_whisperx: bool = True
    diarize: bool = False
    huggingface_token: Optional[str] = None

    steps_to_run: Union[slice, List[int], Tuple[int, ...]] = slice(1, 8) # Default: run all steps (1-7)
    log_level: str = "INFO"
    whisperx_docker_image: str = "echoscribe-whisperx"
    # Add other less frequently changed params here

    def get_full_path(self, folder_or_file_attr: str) -> Path:
        """Helper to get the full path relative to the base_path."""
        return self.base_path / getattr(self, folder_or_file_attr)

# --- Logger Setup ---
log = logging.getLogger("echoscribe.pipeline") # More specific logger name

# --- Helper Functions ---

def should_run_step(step_number: int, steps_to_run: Union[slice, List[int], Tuple[int, ...]]) -> bool:
    """Determine if a 1-based step number should run."""
    max_steps = 8 # Current number of defined steps + 1 for slice behavior
    if isinstance(steps_to_run, slice):
        # Create a range from the slice indices (adjusting for 1-based steps)
        start, stop, step = steps_to_run.indices(max_steps)
        start = max(1, start if start is not None else 1) # Handle None start
        stop = max(start, stop if stop is not None else max_steps) # Handle None stop
        return step_number in range(start, stop, step)
    if isinstance(steps_to_run, (list, tuple)):
        return step_number in steps_to_run
    log.warning(f"Invalid steps_to_run type: {type(steps_to_run)}. Running step {step_number} by default.")
    return True

def _setup_logger(level: str) -> None:
    """Configure the module logger."""
    log_level_upper = level.upper()
    if log_level_upper not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        log.warning(f"Invalid log level '{level}', defaulting to INFO.")
        log_level_upper = "INFO"
    numeric_level = getattr(logging, log_level_upper)

    log.setLevel(numeric_level)
    if not log.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        log.addHandler(handler)
    else: # Update level of existing handler(s)
        for handler in log.handlers:
             handler.setLevel(numeric_level)

def run_whisperx_docker(
    session_path: Path,
    chunks_folder_name: str,
    config: PipelineConfig # Pass config for whisperx image, HF token etc.
    ) -> None:
    """Run WhisperX transcription using Docker (accepts internal config)."""
    log.info("Attempting WhisperX transcription via Docker...")
    chunks_dir = session_path / chunks_folder_name
    if not chunks_dir.is_dir(): raise FileNotFoundError(f"Chunk directory not found: {chunks_dir}")

    flac_files = list(chunks_dir.glob("*.flac"))
    if not flac_files: raise FileNotFoundError(f"No .flac files found in {chunks_dir}")
    log.info(f"Found {len(flac_files)} .flac files for transcription.")

    # --- Command Building ---
    volume_mount = f"{session_path.resolve()}:/app"
    container_chunk_dir = f"/app/{chunks_folder_name}"
    input_args = [f"{container_chunk_dir}/{f.name}" for f in flac_files]
    whisperx_output_dir = f"/app/{config.whisperx_output_folderName}"

    whisperx_args = [
        "whisperx", *input_args,
        "--model", "large-v3", "--language", "fr", # Example args, make configurable?
        "--output_dir", whisperx_output_dir,
        "--output_format", "json"
        # Add other whisperx flags here if needed
    ]
    if config.diarize:
        hf_token = config.huggingface_token or os.environ.get("HF_TOKEN") # Check env again just in case
        if not hf_token: raise ValueError("Diarization enabled, but Hugging Face token missing.")
        log.info("Diarization enabled.")
        whisperx_args.extend(["--diarize", "--hf_token", hf_token])

    docker_command: List[str] = [
        "docker", "run", "--gpus", "all", "--rm", "-v", volume_mount, "--ipc=host",
        config.whisperx_docker_image, # Use image from config
        *whisperx_args
    ]
    # --- Execution ---
    log.info(f"Executing Docker command: {' '.join(docker_command)}")
    try:
        process = subprocess.run(docker_command, capture_output=True, text=True, encoding='utf-8', check=False)
        if process.returncode != 0:
            log.error(f"WhisperX Docker failed (Code: {process.returncode}):\nStderr: {process.stderr}\nStdout: {process.stdout}")
            raise RuntimeError(f"WhisperX Docker execution failed.")
        else:
            log.info("WhisperX Docker command completed successfully.")
            if process.stderr: log.info(f"WhisperX Stderr/Progress:\n{process.stderr}") # Log stderr even on success
            if process.stdout and log.isEnabledFor(logging.DEBUG): log.debug(f"WhisperX Stdout:\n{process.stdout}")
    except FileNotFoundError: log.exception("Docker command not found."); raise
    except Exception as e: log.exception(f"Unexpected error running WhisperX Docker: {e}"); raise


# --- Main Pipeline Function (Interactive Friendly) ---

def run_pipeline(
    # Key parameters directly settable as arguments
    base_path: str = ".",
    steps_to_run: Union[slice, List[int], Tuple[int, ...]] = slice(1, None), # Default slice includes step 7
    log_level: str = "INFO",
    run_whisperx: bool = True,
    diarize: bool = False,
    use_low_ram: bool = True,
    # Allow passing other config values directly if needed often
    input_audio_folderName: Optional[str] = None,
    # Optional config object for advanced/bulk settings
    config_obj: Optional[PipelineConfig] = None,
    # Allow overriding specific settings via kwargs
    **kwargs: Any
    ) -> None:
    """
    Run the audio transcription pipeline with flexible configuration.

    Allows setting key parameters directly or passing a full config object.
    Direct arguments override values in config_obj or defaults.

    Args:
        base_path: Base directory for the session. Defaults to current directory.
        steps_to_run: Slice, list, or tuple defining 1-based steps to run.
                      Defaults to slice(1, None) which runs all steps 1 through 7.
                      Example: steps_to_run=[1, 3, 7] or steps_to_run=slice(1, 4)
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        run_whisperx: Whether to run the WhisperX Docker step automatically.
        diarize: Enable speaker diarization (requires Hugging Face token).
        use_low_ram: Use low RAM mode for audio chunking/silence detection.
        input_audio_folderName: Optional override for the input tracks folder name.
        config_obj: An optional pre-configured PipelineConfig object.
        **kwargs: Additional configuration overrides (e.g., final_folderName="output").
                  Keys must match attributes in PipelineConfig.
    """
    _setup_logger(log_level) # Setup logger first with requested level
    log.info(f"Initializing echoscribe pipeline (v{__version__ if '__version__' in globals() else 'unknown'})...") # Add version if available

    # --- Configuration Merging ---
    # Start with default config
    current_config_dict = vars(PipelineConfig())

    # Update with config_obj if provided
    if config_obj is not None:
        if isinstance(config_obj, PipelineConfig):
            current_config_dict.update(vars(config_obj))
            log.info("Loaded base settings from provided config_obj.")
        else:
            log.warning("Invalid config_obj provided, using defaults.")

    # Update with direct arguments (highest precedence)
    # Use a dict to map direct args to config fields for clarity
    direct_args = {
        "base_path": base_path,
        "steps_to_run": steps_to_run,
        "log_level": log_level,
        "run_whisperx": run_whisperx,
        "diarize": diarize,
        "use_low_ram": use_low_ram,
        "input_audio_folderName": input_audio_folderName,
    }
    # Filter out None values from direct args so they don't override defaults unintentionally
    valid_direct_args = {k: v for k, v in direct_args.items() if v is not None}
    current_config_dict.update(valid_direct_args)

    # Update with kwargs (carefully check keys match PipelineConfig fields)
    valid_kwargs = {}
    config_fields = PipelineConfig.__annotations__.keys()
    for key, value in kwargs.items():
        if key in config_fields:
            valid_kwargs[key] = value
        else:
            log.warning(f"Ignoring unknown configuration kwarg: {key}")
    current_config_dict.update(valid_kwargs)

    # Handle base_path conversion to Path object now
    try:
        current_config_dict["base_path"] = Path(current_config_dict["base_path"]).resolve()
    except (TypeError, ValueError) as e:
        log.exception(f"Invalid base_path '{current_config_dict['base_path']}' provided.")
        raise ValueError(f"Invalid base_path: {e}") from e

    # Create the final internal config object
    try:
        config = PipelineConfig(**current_config_dict)
        config.base_path.mkdir(parents=True, exist_ok=True) # Ensure base path exists
    except TypeError as e:
        log.exception(f"Failed to create PipelineConfig from provided arguments/kwargs: {e}")
        raise ValueError(f"Configuration error: {e}") from e
    except OSError as e:
        log.exception(f"Failed to create base directory: {config.base_path}")
        raise IOError(f"Could not create base directory {config.base_path}") from e

    log.info(f"Effective configuration: {config}") # Log the final merged config

    # Define paths using the final config and helper
    input_audio_folder = config.get_full_path("input_audio_folderName")
    chunked_audio_folder = config.get_full_path("chunks_audio_folderName")
    whisperx_output_folder = config.get_full_path("whisperx_output_folderName")
    corrected_json_folder = config.get_full_path("corrected_json_folderName")
    srt_folder = config.get_full_path("srt_folderName")
    final_folder = config.get_full_path("final_folderName")
    merged_file_path = final_folder / config.merged_audio_filename
    cut_points_file_path = final_folder / config.cut_points_filename
    merged_srt_path = final_folder / config.merged_transcript_filename
    cleaned_srt_path = final_folder / config.cleaned_transcript_filename
    final_script_path = final_folder / config.final_script_filename

    # Create final output folder early
    try: final_folder.mkdir(parents=True, exist_ok=True)
    except OSError as e: log.exception(f"Failed to create final output directory: {final_folder}"); raise IOError(f"Could not create final dir {final_folder}") from e

    steps_executed: List[int] = []
    pipeline_successful = True

    # --- Step Execution ---
    # Step 1: Chunking Audio
    if should_run_step(1, config.steps_to_run):
        log.info("=== Step 1: Chunking audio tracks ===")
        try:
            if not input_audio_folder.is_dir(): raise FileNotFoundError(f"Input audio dir not found: {input_audio_folder}")
            # **ASSUMPTION**: chunk_audio accepts Path objects and use_low_ram bool
            chunk_audio(input_folder=input_audio_folder, output_dir=chunked_audio_folder,
                        merged_file=merged_file_path, cut_points_file=cut_points_file_path,
                        use_low_ram=config.use_low_ram)
            steps_executed.append(1); log.info("--- Step 1 completed ---")
        except Exception as e:
            log.exception("!!! Step 1 FAILED: Chunking audio !!!"); pipeline_successful = False
        if not pipeline_successful: return # Stop if critical step fails

    # Step 2: WhisperX Transcription
    if should_run_step(2, config.steps_to_run):
        log.info("=== Step 2: WhisperX transcription ===")
        try:
            if config.run_whisperx:
                run_whisperx_docker(config.base_path, config.chunks_audio_folderName, config)
            else:
                log.warning("run_whisperx=False. Manual transcription required.")
                log.warning(f"Ensure JSON outputs from '{chunked_audio_folder.name}/' are placed in '{whisperx_output_folder.name}/'.")
                # Consider removing input() for library use or making it optional
                # input("Press Enter to continue once manual transcription is complete...")
                if not whisperx_output_folder.is_dir() or not any(whisperx_output_folder.iterdir()):
                    raise FileNotFoundError(f"WhisperX output dir '{whisperx_output_folder}' empty/missing after manual step.")
            steps_executed.append(2); log.info("--- Step 2 completed ---")
        except Exception as e:
            log.exception("!!! Step 2 FAILED: WhisperX transcription !!!"); pipeline_successful = False
        if not pipeline_successful: return

    # Step 3: Correcting WhisperX Output
    if should_run_step(3, config.steps_to_run):
        log.info("=== Step 3: Correcting WhisperX output ===")
        try:
            if not whisperx_output_folder.is_dir(): raise FileNotFoundError(f"Input dir not found: {whisperx_output_folder}")
            # **ASSUMPTION**: correct_whisperx_outputs accepts string paths and log_level
            correct_whisperx_outputs(input_dir=str(whisperx_output_folder), output_dir=str(corrected_json_folder),
                                     log_level=config.log_level) # Pass log level for consistency
            steps_executed.append(3); log.info("--- Step 3 completed ---")
        except Exception as e:
            log.exception("!!! Step 3 FAILED: Correcting WhisperX output !!!"); pipeline_successful = False
        if not pipeline_successful: return

    # Step 4: Converting Corrected JSON to SRT
    if should_run_step(4, config.steps_to_run):
        log.info("=== Step 4: Converting corrected JSON to SRT ===")
        try:
            if not corrected_json_folder.is_dir(): raise FileNotFoundError(f"Input dir not found: {corrected_json_folder}")
            # **ASSUMPTION**: convert_json_folder_to_srt accepts string paths
            convert_json_folder_to_srt(input_dir=str(corrected_json_folder), output_dir=str(srt_folder))
            steps_executed.append(4); log.info("--- Step 4 completed ---")
        except Exception as e:
            log.exception("!!! Step 4 FAILED: Converting JSON to SRT !!!"); pipeline_successful = False
        if not pipeline_successful: return

    # Step 5: Merging SRT files by chunk
    if should_run_step(5, config.steps_to_run):
        log.info("=== Step 5: Merging SRT files by chunk ===")
        try:
            if not srt_folder.is_dir(): raise FileNotFoundError(f"Input dir not found: {srt_folder}")
            # **ASSUMPTION**: merge_srt_by_chunk accepts string paths/filenames
            merge_srt_by_chunk(srt_folder=str(srt_folder), output_dir=str(final_folder),
                               output_filename=config.merged_transcript_filename)
            steps_executed.append(5); log.info("--- Step 5 completed ---")
        except Exception as e:
            log.exception("!!! Step 5 FAILED: Merging SRT files !!!"); pipeline_successful = False
        if not pipeline_successful: return

    # Step 6: Merging successive speaker entries
    if should_run_step(6, config.steps_to_run):
        log.info("=== Step 6: Merging successive speaker entries ===")
        try:
            if not merged_srt_path.is_file(): raise FileNotFoundError(f"Input file not found: {merged_srt_path}")
            # **ASSUMPTION**: merge_speaker_entries accepts string paths
            merge_speaker_entries(input_srt_file=str(merged_srt_path), output_srt_file=str(cleaned_srt_path))
            steps_executed.append(6); log.info("--- Step 6 completed ---")
        except Exception as e:
            log.exception("!!! Step 6 FAILED: Merging speaker entries !!!"); pipeline_successful = False
        if not pipeline_successful: return

    # Step 7: Creating final readable script
    if should_run_step(7, config.steps_to_run):
        log.info("=== Step 7: Creating final readable script ===")
        try:
            if not cleaned_srt_path.is_file(): raise FileNotFoundError(f"Input file not found: {cleaned_srt_path}")
            # **ASSUMPTION**: srt_to_script accepts string paths
            srt_to_script(input_srt_path=str(cleaned_srt_path), output_script_path=str(final_script_path))
            steps_executed.append(7); log.info("--- Step 7 completed ---")
        except Exception as e:
            log.exception("!!! Step 7 FAILED: Creating final script !!!"); pipeline_successful = False
        if not pipeline_successful: return

    # --- Final Summary ---
    log.info("========================================")
    if pipeline_successful:
        all_steps_default = list(range(1, 8)) # Steps 1 to 7
        ran_all_requested = True
        if isinstance(config.steps_to_run, slice):
            start, stop, step = config.steps_to_run.indices(8)
            requested_steps = list(range(max(1, start), stop, step))
            if set(steps_executed) != set(requested_steps): ran_all_requested = False
        elif isinstance(config.steps_to_run, (list, tuple)):
             if set(steps_executed) != set(config.steps_to_run): ran_all_requested = False

        if ran_all_requested and steps_executed == all_steps_default:
            log.info(f"✅ Full pipeline finished successfully! Outputs available in: {final_folder}")
        elif ran_all_requested:
            log.info(f"✅ Requested pipeline steps {steps_executed} finished successfully! Outputs available in: {final_folder}")
        else:
             log.warning(f"Pipeline finished, but steps executed ({steps_executed}) may not match requested ({config.steps_to_run}). Outputs in: {final_folder}")
    else:
        log.error(f"Pipeline finished with errors after executing steps: {steps_executed}. Check logs for details.")
        # Consider raising an exception to signal failure to the caller
        # raise RuntimeError("Pipeline failed during execution.")


# Example of interactive usage
# if __name__ == "__main__":
#     print("Running pipeline example...")
#     # Example 1: Default run in current directory
#     # run_pipeline()

#     # Example 2: Specify base path and only run steps 1-3 with DEBUG logging
#     # run_pipeline(base_path="my_test_session", steps_to_run=slice(1, 4), log_level="DEBUG")

#     # Example 3: Run all steps but enable diarization and override final folder name
#     # run_pipeline(base_path="another_session", diarize=True, final_folderName="final_diarized_output")

#     # Example 4: Run only step 3 (correction)
#     # run_pipeline(base_path="session_needs_correction", steps_to_run=[3])