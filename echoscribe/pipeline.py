# echoscribe/pipeline.py

import os
import subprocess
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Optional, Union, List, Tuple, Sequence, Dict, Any
import select

# --- Imports ---
from .modules.audio_chunker import chunk_audio, ChunkingConfig, DEFAULT_CHUNKING_CONFIG
from .modules.whisperx_corrector_core import correct_whisperx_outputs, WhisperXCorrectorConfig, DEFAULT_CONFIG as DEFAULT_CORRECTOR_CONFIG
from .modules.convert_json_to_srt import convert_json_folder_to_srt
from .modules.merge_srt_by_chunk import merge_srt_by_chunk, SRTMergeConfig, DEFAULT_MERGE_CONFIG
from .modules.merge_speaker_entries import merge_speaker_entries
from .modules.convert_srt_to_script import srt_to_script

# --- Configuration Dataclass ---

@dataclass(frozen=True)
class PipelineConfig:
    """Internal configuration settings for the audio processing pipeline."""
    # Paths and Names
    base_path: Path = field(default_factory=Path.cwd)
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

    # Core Flags
    use_low_ram: bool = True
    run_whisperx: bool = True
    diarize: bool = False
    huggingface_token: Optional[str] = None

    # Execution Control
    steps_to_run: Union[slice, List[int], Tuple[int, ...]] = slice(1, 8) # Default: run steps 1-7
    log_level: str = "INFO"

    # External Tools Config
    whisperx_docker_image: str = "echoscribe-whisperx"

    # Module Specific Configs (using defaults, extend later if needed)
    chunking_config: ChunkingConfig = field(default_factory=lambda: DEFAULT_CHUNKING_CONFIG)
    corrector_config: WhisperXCorrectorConfig = field(default_factory=lambda: DEFAULT_CORRECTOR_CONFIG)
    srtmerge_config: SRTMergeConfig = field(default_factory=lambda: DEFAULT_MERGE_CONFIG)

    def get_full_path(self, folder_or_file_attr: str) -> Path:
        """Helper to get the full path relative to the base_path."""
        return self.base_path / getattr(self, folder_or_file_attr)

# --- Logger Setup ---
# Use a more descriptive name if part of a larger package
log = logging.getLogger("echoscribe.pipeline")

# --- Helper Functions ---

def should_run_step(step_number: int, steps_to_run: Union[slice, List[int], Tuple[int, ...]]) -> bool:
    """Determine if a 1-based step number should run."""
    max_steps = 8 # Steps 1-7 exist, +1 for slice range
    if isinstance(steps_to_run, slice):
        start, stop, step = steps_to_run.indices(max_steps)
        start = max(1, start if start is not None else 1)
        stop = max(start, stop if stop is not None else max_steps)
        return step_number in range(start, stop, step)
    if isinstance(steps_to_run, (list, tuple)):
        return step_number in steps_to_run
    log.warning(f"Invalid steps_to_run type: {type(steps_to_run)}. Running step {step_number} by default.")
    return True

def _setup_logger(level: str) -> None:
    """Configure the root logger for the application."""
    log_level_upper = level.upper()
    if log_level_upper not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        log.warning(f"Invalid log level '{level}', defaulting to INFO.")
        log_level_upper = "INFO"
    numeric_level = getattr(logging, log_level_upper)

    # --- Configure ROOT Logger ---
    root_logger = logging.getLogger() # Get the root logger
    root_logger.setLevel(numeric_level) # Set level for all handlers unless overridden

    # Check if a suitable handler already exists on the root logger
    handler_exists = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in root_logger.handlers)

    if not handler_exists:
        # Add a handler logging to STDOUT
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        # Set the handler level explicitly too (usually inherits from logger, but good practice)
        handler.setLevel(numeric_level)
        root_logger.addHandler(handler)
        log.info(f"Configured root logger with handler level {log_level_upper}") # Log this once
    else:
        # Ensure existing handlers respect the requested level
        for handler in root_logger.handlers:
             # Optionally update level of existing handlers if needed
             # handler.setLevel(numeric_level)
             pass # Often handlers retain their own configured level

    # Individual modules will use log = logging.getLogger(__name__)
    # and their messages will propagate to the root handler setup here.


def run_whisperx_docker(
    session_path: Path,
    chunks_folder_name: str,
    config: PipelineConfig
    ) -> None:
    """Run WhisperX Docker, streaming output in real-time."""
    log.info("Attempting WhisperX transcription via Docker (streaming output)...")
    # ... (Input validation: chunks_dir, flac_files - same as before) ...
    chunks_dir = session_path / chunks_folder_name
    if not chunks_dir.is_dir(): raise FileNotFoundError(f"Chunk directory not found: {chunks_dir}")
    flac_files = list(chunks_dir.glob("*.flac"))
    if not flac_files: raise FileNotFoundError(f"No .flac files found in {chunks_dir}")
    log.info(f"Found {len(flac_files)} .flac files for transcription.")

    # ... (Command building: volume_mount, whisperx_args, docker_command - same as before) ...
    volume_mount = f"{session_path.resolve()}:/app"
    container_chunk_dir = f"/app/{chunks_folder_name}"
    input_args = [f"{container_chunk_dir}/{f.name}" for f in flac_files]
    whisperx_output_dir = f"/app/{config.whisperx_output_folderName}"
    whisperx_args = ["whisperx", *input_args, "--model", "large-v3", "--language", "fr",
                     "--output_dir", whisperx_output_dir, "--output_format", "json"]
    if config.diarize:
        hf_token = config.huggingface_token or os.environ.get("HF_TOKEN")
        if not hf_token: raise ValueError("Diarization enabled, but Hugging Face token missing.")
        log.info("Diarization enabled.")
        whisperx_args.extend(["--diarize", "--hf_token", hf_token])
    docker_command: List[str] = ["docker", "run", "--gpus", "all", "--rm", "-v", volume_mount,
                                 "--ipc=host", config.whisperx_docker_image, *whisperx_args]

    log.info(f"Executing Docker command: {' '.join(docker_command)}")

    # --- Use Popen to stream output ---
    process = None
    try:
        process = subprocess.Popen(
            docker_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            bufsize=1 # Line buffered
        )

        # Read stdout and stderr streams line by line
        # Use select to avoid blocking indefinitely on one stream if the other has data
        streams = {process.stdout: sys.stdout, process.stderr: sys.stderr}
        while process.poll() is None: # While process is running
            # Check which streams are ready for reading
            readable, _, _ = select.select(streams.keys(), [], [], 0.1) # Timeout 0.1s
            for stream in readable:
                 line = stream.readline()
                 if line:
                      # Write docker output directly to corresponding system stream
                      # Add a prefix to distinguish docker output
                      prefix = "[Docker STDOUT] " if stream is process.stdout else "[Docker STDERR] "
                      streams[stream].write(prefix + line)
                      streams[stream].flush()

        # Process finished, read any remaining output
        stdout_rem, stderr_rem = process.communicate()
        if stdout_rem:
            for line in stdout_rem.splitlines():
                 sys.stdout.write("[Docker STDOUT] " + line + "\n")
            sys.stdout.flush()
        if stderr_rem:
             for line in stderr_rem.splitlines():
                 sys.stderr.write("[Docker STDERR] " + line + "\n")
             sys.stderr.flush()

        # Check final return code
        retcode = process.returncode
        if retcode != 0:
            log.error(f"WhisperX Docker command failed with exit code {retcode}")
            raise RuntimeError(f"WhisperX Docker execution failed (exit code {retcode}).")
        else:
            log.info("WhisperX Docker command completed successfully.")

    except FileNotFoundError: log.exception("Docker command not found."); raise
    except Exception as e: log.exception(f"Unexpected error running WhisperX Docker: {e}"); raise
    finally:
         # Ensure streams are closed even if errors occur
         if process:
              if process.stdout: process.stdout.close()
              if process.stderr: process.stderr.close()
              # Optionally wait again to ensure process is fully cleaned up
              try: process.wait(timeout=1)
              except subprocess.TimeoutExpired: process.kill()


# --- Main Pipeline Function (Interactive Friendly) ---

def run_pipeline(
    base_path: str = ".",
    steps_to_run: Union[slice, List[int], Tuple[int, ...]] = slice(1, 8), # Slice includes 7 now
    log_level: str = "INFO",
    run_whisperx: bool = True,
    diarize: bool = False,
    use_low_ram: bool = True,
    input_audio_folderName: Optional[str] = None,
    config_obj: Optional[PipelineConfig] = None,
    **kwargs: Any
    ) -> None:
    """
    Run the audio transcription pipeline with flexible configuration.

    Allows setting key parameters directly or passing a full config object.
    Direct arguments override values in config_obj or defaults.

    Args:
        base_path: Base directory for the session.
        steps_to_run: 1-based steps to run (slice, list, or tuple). Default runs 1-7.
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        run_whisperx: Whether to run the WhisperX Docker step automatically.
        diarize: Enable speaker diarization (requires Hugging Face token).
        use_low_ram: Use low RAM mode for audio chunking.
        input_audio_folderName: Optional override for the input tracks folder name.
        config_obj: An optional pre-configured PipelineConfig object.
        **kwargs: Additional configuration overrides matching PipelineConfig attributes.
    """
    _setup_logger(log_level)
    log.info(f"Initializing echoscribe pipeline (v{__version__ if '__version__' in globals() else 'unknown'})...")

    # --- Configuration Merging ---
    try:
        # Start with defaults from the dataclass definition
        base_config_dict = vars(PipelineConfig())

        # Layer 1: Update with provided config_obj if valid
        if config_obj is not None:
            if isinstance(config_obj, PipelineConfig):
                base_config_dict.update(vars(config_obj))
                log.info("Loaded base settings from provided config_obj.")
            else:
                log.warning("Invalid config_obj type provided, ignoring.")

        # Layer 2: Update with direct arguments (filter Nones)
        direct_args = {"base_path": base_path, "steps_to_run": steps_to_run, "log_level": log_level,
                       "run_whisperx": run_whisperx, "diarize": diarize, "use_low_ram": use_low_ram,
                       "input_audio_folderName": input_audio_folderName}
        valid_direct_args = {k: v for k, v in direct_args.items() if v is not None}
        base_config_dict.update(valid_direct_args)

        # Layer 3: Update with valid kwargs
        config_fields = PipelineConfig.__annotations__.keys()
        valid_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        if unknown_kwargs := set(kwargs) - set(valid_kwargs):
            log.warning(f"Ignoring unknown configuration kwargs: {unknown_kwargs}")
        base_config_dict.update(valid_kwargs)

        # Convert base_path to resolved Path object *before* creating final config
        base_path_str = base_config_dict["base_path"]
        if not isinstance(base_path_str, (str, Path)): raise TypeError(f"base_path must be a string or Path, got {type(base_path_str)}")
        base_config_dict["base_path"] = Path(base_path_str).resolve()

        # Create the final config instance
        config = PipelineConfig(**base_config_dict)
        config.base_path.mkdir(parents=True, exist_ok=True) # Ensure base path exists

    except (TypeError, ValueError, OSError, Exception) as e:
        log.exception(f"Failed during configuration setup: {e}")
        # Re-raise as a generic configuration error or specific type if preferred
        raise ValueError(f"Pipeline configuration failed: {e}") from e

    log.info(f"Effective configuration: {config}")

    # Define paths using the final config
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
    except OSError as e: log.exception(f"Failed creating final output directory: {final_folder}"); raise IOError(f"Could not create final dir {final_folder}") from e

    # --- Input/Output Path Collision Check ---
    if input_audio_folder.resolve() == final_folder.resolve():
         log.warning("Input audio folder and final output folder resolve to the same path. This might cause unexpected behavior.")
    # Add more specific checks if needed, e.g., output_dir for a step == input_dir for next step

    steps_executed: List[int] = []
    pipeline_successful = True

    # --- Step Execution ---
    # Step 1: Chunking Audio
    if should_run_step(1, config.steps_to_run):
        log.info("=== Step 1: Chunking audio tracks ===")
        try:
            if not input_audio_folder.is_dir(): raise FileNotFoundError(f"Input audio dir not found: {input_audio_folder}")
            # Pass Path objects and relevant config piece
            chunk_audio(input_folder=input_audio_folder, output_dir=chunked_audio_folder,
                        merged_file=merged_file_path, cut_points_file=cut_points_file_path,
                        use_low_ram=config.use_low_ram, config=config.chunking_config) # Pass specific config
            steps_executed.append(1); log.info("--- Step 1 completed ---")
        except Exception as e: log.exception("!!! Step 1 FAILED !!!"); pipeline_successful = False
        if not pipeline_successful: return

    # Step 2: WhisperX Transcription
    if should_run_step(2, config.steps_to_run):
        log.info("=== Step 2: WhisperX transcription ===")
        try:
            if config.run_whisperx:
                run_whisperx_docker(config.base_path, config.chunks_audio_folderName, config)
            else: # Manual step
                log.warning("run_whisperx=False. Manual transcription required.")
                log.warning(f"Ensure JSON outputs from '{chunked_audio_folder.name}/' are in '{whisperx_output_folder.name}/'.")
                # input("Press Enter to continue...") # Commented out for non-interactive use potential
                if not whisperx_output_folder.is_dir() or not any(whisperx_output_folder.iterdir()):
                    raise FileNotFoundError(f"WhisperX output dir '{whisperx_output_folder}' empty/missing after manual step.")
            steps_executed.append(2); log.info("--- Step 2 completed ---")
        except Exception as e: log.exception("!!! Step 2 FAILED !!!"); pipeline_successful = False
        if not pipeline_successful: return

    # Step 3: Correcting WhisperX Output
    if should_run_step(3, config.steps_to_run):
        log.info("=== Step 3: Correcting WhisperX output ===")
        try:
            if not whisperx_output_folder.is_dir(): raise FileNotFoundError(f"Input dir not found: {whisperx_output_folder}")
            # Pass Path objects now, assuming adapter accepts them (should be updated if not)
            correct_whisperx_outputs(input_dir=whisperx_output_folder, output_dir=corrected_json_folder)
            steps_executed.append(3); log.info("--- Step 3 completed ---")
        except Exception as e: log.exception("!!! Step 3 FAILED !!!"); pipeline_successful = False
        if not pipeline_successful: return

    # Step 4: Converting Corrected JSON to SRT
    if should_run_step(4, config.steps_to_run):
        log.info("=== Step 4: Converting corrected JSON to SRT ===")
        try:
            if not corrected_json_folder.is_dir(): raise FileNotFoundError(f"Input dir not found: {corrected_json_folder}")
            # Pass Path objects
            convert_json_folder_to_srt(input_dir=corrected_json_folder, output_dir=srt_folder)
            steps_executed.append(4); log.info("--- Step 4 completed ---")
        except Exception as e: log.exception("!!! Step 4 FAILED !!!"); pipeline_successful = False
        if not pipeline_successful: return

    # Step 5: Merging SRT files by chunk
    if should_run_step(5, config.steps_to_run):
        log.info("=== Step 5: Merging SRT files by chunk ===")
        try:
            if not srt_folder.is_dir(): raise FileNotFoundError(f"Input dir not found: {srt_folder}")
            # Pass Path objects and relevant config piece
            merge_srt_by_chunk(srt_folder=srt_folder, output_dir=final_folder,
                               output_filename=config.merged_transcript_filename,
                               config=config.srtmerge_config) # Pass specific config
            steps_executed.append(5); log.info("--- Step 5 completed ---")
        except Exception as e: log.exception("!!! Step 5 FAILED !!!"); pipeline_successful = False
        if not pipeline_successful: return

    # Step 6: Merging successive speaker entries
    if should_run_step(6, config.steps_to_run):
        log.info("=== Step 6: Merging successive speaker entries ===")
        try:
            if not merged_srt_path.is_file(): raise FileNotFoundError(f"Input file not found: {merged_srt_path}")
            # Pass Path objects
            merge_speaker_entries(input_srt_file=merged_srt_path, output_srt_file=cleaned_srt_path)
            steps_executed.append(6); log.info("--- Step 6 completed ---")
        except Exception as e: log.exception("!!! Step 6 FAILED !!!"); pipeline_successful = False
        if not pipeline_successful: return

    # Step 7: Creating final readable script
    if should_run_step(7, config.steps_to_run):
        log.info("=== Step 7: Creating final readable script ===")
        try:
            if not cleaned_srt_path.is_file(): raise FileNotFoundError(f"Input file not found: {cleaned_srt_path}")
            # Pass Path objects
            srt_to_script(input_srt_path=cleaned_srt_path, output_script_path=final_script_path)
            steps_executed.append(7); log.info("--- Step 7 completed ---")
        except Exception as e: log.exception("!!! Step 7 FAILED !!!"); pipeline_successful = False
        if not pipeline_successful: return

    # --- Final Summary ---
    log.info("========================================")
    if pipeline_successful:
        all_steps_default = list(range(1, 8)); ran_all_requested = True
        # Check if executed matches requested
        if isinstance(config.steps_to_run, slice):
            start, stop, step = config.steps_to_run.indices(8); requested_steps = list(range(max(1, start), stop, step))
            if set(steps_executed) != set(requested_steps): ran_all_requested = False
        elif isinstance(config.steps_to_run, (list, tuple)):
             if set(steps_executed) != set(config.steps_to_run): ran_all_requested = False
        # Log appropriate message
        if ran_all_requested and steps_executed == all_steps_default: log.info(f"✅ Full pipeline finished successfully! Outputs: {final_folder}")
        elif ran_all_requested: log.info(f"✅ Requested steps {steps_executed} finished successfully! Outputs: {final_folder}")
        else: log.warning(f"Pipeline finished, but steps executed ({steps_executed}) may not match requested ({config.steps_to_run}). Outputs: {final_folder}")
    else:
        log.error(f"Pipeline finished with errors after executing steps: {steps_executed}. Check logs.")
        raise RuntimeError("Pipeline failed during execution.")

# Example of interactive usage
# if __name__ == "__main__": run_pipeline(...)