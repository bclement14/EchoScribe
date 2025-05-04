# echoscribe/modules/merge_speaker_entries.py

import logging
import re
from pathlib import Path
import srt # External library: https://pypi.org/project/srt/
from typing import List, Optional, Tuple, Union, Dict

# --- Module Logger ---
log = logging.getLogger(__name__)

# --- Constants ---
SPEAKER_TAG_REGEX = re.compile(r'^\[([^\]]+)\]\s*(.*)') # Group 1=Speaker, Group 2=Text

# --- Helper Function ---
def _extract_speaker(content: str) -> Tuple[Optional[str], str]:
    """Extracts the speaker name and the remaining content."""
    match = SPEAKER_TAG_REGEX.match(content)
    if match:
        speaker = match.group(1).strip()
        text = match.group(2).strip()
        # Return speaker name without brackets for comparison
        return speaker, text
    else:
        return None, content.strip()

# --- Main Function ---
def merge_speaker_entries(
    input_srt_file: Union[str, Path],
    output_srt_file: Union[str, Path]
    ) -> None:
    """
    Reads an SRT file, merges consecutive entries from the same speaker,
    keeping the speaker tag, and writes the cleaned SRT file.

    Args:
        input_srt_file: Path to the input merged SRT file (expecting tags like [Speaker]).
        output_srt_file: Path to save the cleaned SRT file (will retain tags).

    Raises:
        FileNotFoundError, IOError, srt.SRTParseError, OSError as before.
    """
    input_path = Path(input_srt_file)
    output_path = Path(output_srt_file)

    log.info(f"Merging consecutive speaker entries in '{input_path.name}'...")
    log.info(f"Output will be saved to '{output_path}' (retaining speaker tags)")

    if not input_path.is_file():
        log.error(f"Input SRT file not found: {input_path}")
        raise FileNotFoundError(f"Input SRT file not found: {input_path}")

    try:
        content = input_path.read_text(encoding='utf-8')
        subtitles: List[srt.Subtitle] = list(srt.parse(content))
        log.info(f"Parsed {len(subtitles)} subtitles from input file.")
    except Exception as e: # Catch parsing/reading errors
        log.exception(f"Failed to read/parse input SRT file {input_path}:")
        raise # Re-raise original exception

    merged_subtitles: List[srt.Subtitle] = []
    if not subtitles:
        log.warning("Input SRT file contains no subtitles. Output file will be empty.")
        # Still write empty file for consistency
    else:
        # Use a temporary dictionary to store current merged data
        current_merge_data: Dict[str, Any] = {}
        merges_count = 0

        for i, current_sub in enumerate(subtitles):
            current_speaker, current_text = _extract_speaker(current_sub.content)

            if not current_merge_data: # First subtitle
                current_merge_data = {
                    "speaker": current_speaker,
                    "text": current_text,
                    "start": current_sub.start,
                    "end": current_sub.end,
                    "index": 1 # Start SRT index at 1
                }
            elif current_merge_data["speaker"] is not None and current_merge_data["speaker"] == current_speaker:
                # --- Same Speaker: Merge ---
                # Append text with a space
                separator = " " if current_merge_data["text"] else ""
                current_merge_data["text"] += separator + current_text
                # Update end time
                current_merge_data["end"] = current_sub.end
                merges_count += 1
                if log.isEnabledFor(logging.DEBUG):
                     log.debug(f"Merged entry for speaker '{current_speaker}' ending at {current_sub.end}")
            else:
                # --- Different Speaker or Untagged: Finalize previous and start new ---
                # Create Subtitle object from previous merged data
                prev_speaker = current_merge_data["speaker"]
                prev_text = current_merge_data["text"]
                # *** CORRECTED: Prepend speaker tag back ***
                final_content = f"[{prev_speaker}] {prev_text}" if prev_speaker else prev_text
                merged_subtitles.append(
                    srt.Subtitle(
                        index=current_merge_data["index"],
                        start=current_merge_data["start"],
                        end=current_merge_data["end"],
                        content=final_content
                    )
                )
                # Start new entry
                current_merge_data = {
                    "speaker": current_speaker,
                    "text": current_text,
                    "start": current_sub.start,
                    "end": current_sub.end,
                    "index": len(merged_subtitles) + 1 # Increment index
                }
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(f"Starting new entry for speaker: {current_speaker}")

        # Append the very last processed entry after the loop
        if current_merge_data:
            prev_speaker = current_merge_data["speaker"]
            prev_text = current_merge_data["text"]
            # *** CORRECTED: Prepend speaker tag back ***
            final_content = f"[{prev_speaker}] {prev_text}" if prev_speaker else prev_text
            merged_subtitles.append(
                srt.Subtitle(
                    index=current_merge_data["index"],
                    start=current_merge_data["start"],
                    end=current_merge_data["end"],
                    content=final_content
                )
            )

        log.info(f"Merge complete. Performed {merges_count} merges, resulting in {len(merged_subtitles)} final subtitles.")

    # --- Writing Output ---
    try:
        log.info(f"Writing {len(merged_subtitles)} merged subtitles to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_srt_content = srt.compose(merged_subtitles)
        output_path.write_text(final_srt_content, encoding='utf-8')
        log.info("Merged speaker entries saved successfully (with tags retained).")
    except Exception as e: # Catch writing/compose errors
        log.exception(f"Failed to write output SRT file: {output_path}")
        raise IOError(f"Failed to write output SRT file {output_path}") from e