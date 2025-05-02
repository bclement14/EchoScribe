# echoscribe/modules/__init__.py

# Import submodules (keep namespace access)
from . import treat_flac_tracks
from . import correct_json_output
from . import convert_json_to_srt
from . import merge_srt_by_chunk
from . import merge_speaker_entries
from . import convert_srt_to_script
from . import combined_whisperx_corrector_r as cwcr

# Re-export key functions for easy access
from .treat_flac_tracks import chunk_audio
from .correct_json_output import correct_whisperx_outputs
from .convert_json_to_srt import convert_json_folder_to_srt
from .merge_srt_by_chunk import merge_srt_by_chunk
from .merge_speaker_entries import merge_speaker_entries
from .convert_srt_to_script import srt_to_script

__all__ = [
    "chunk_audio",
    "correct_whisperx_outputs",
    "convert_json_folder_to_srt",
    "merge_srt_by_chunk",
    "merge_speaker_entries",
    "srt_to_script",
]
