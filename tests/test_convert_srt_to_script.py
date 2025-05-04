# tests/test_convert_srt_to_script.py

import pytest
from pathlib import Path
import logging # Import logging to potentially capture logs if needed

# Import the function to test
from echoscribe.modules.convert_srt_to_script import srt_to_script

# Define sample SRT content
SAMPLE_SRT_CONTENT = """1
00:00:01,000 --> 00:00:03,500
[Speaker A] Hello there.
This is line two.

2
00:00:04,100 --> 00:00:05,900
[Speaker B] Hi! How are you?

3
00:00:06,000 --> 00:00:07,000
Okay.

4
00:00:08,000 --> 00:00:09,500
[Speaker A] I'm doing well, thanks.
Multi-line again.

"""

EMPTY_SRT_CONTENT = ""

MALFORMED_SRT_CONTENT = """1
00:00:01,000 --> 00:00:03,500
[Speaker A] First line okay.

This block has no timestamp or index.

3
00:00:06,000 --> 00:00:07,000
Okay.

4
00:00:08,000 --> 00:00:09,500
""" # Missing text line

EXPECTED_SCRIPT_OUTPUT = """[Speaker A] Hello there. This is line two.

[Speaker B] Hi! How are you?

Okay.

[Speaker A] I'm doing well, thanks. Multi-line again."""

# --- Test Functions ---

def test_srt_to_script_standard(tmp_path: Path):
    """Test conversion with a standard, valid SRT file."""
    input_srt = tmp_path / "input.srt"
    output_script = tmp_path / "output.txt"
    input_srt.write_text(SAMPLE_SRT_CONTENT, encoding="utf-8")
    srt_to_script(input_srt_path=input_srt, output_script_path=output_script)
    assert output_script.exists()
    # --- Assert against EXPECTED_SCRIPT_OUTPUT ---
    assert output_script.read_text(encoding="utf-8") == EXPECTED_SCRIPT_OUTPUT

def test_srt_to_script_output_dir_creation(tmp_path: Path):
    """Test that the output directory is created if it doesn't exist."""
    input_srt = tmp_path / "input.srt"
    output_dir = tmp_path / "output_subdir"
    output_script = output_dir / "output.txt"
    input_srt.write_text(SAMPLE_SRT_CONTENT, encoding="utf-8")

    # Ensure output dir doesn't exist initially
    assert not output_dir.exists()

    srt_to_script(input_srt_path=input_srt, output_script_path=output_script)

    assert output_dir.exists()
    assert output_dir.is_dir()
    assert output_script.exists()
    assert output_script.read_text(encoding="utf-8") == EXPECTED_SCRIPT_OUTPUT

def test_srt_to_script_empty_file(tmp_path: Path):
    """Test conversion with an empty input SRT file."""
    input_srt = tmp_path / "input_empty.srt"
    output_script = tmp_path / "output_empty.txt"
    input_srt.write_text(EMPTY_SRT_CONTENT, encoding="utf-8")

    srt_to_script(input_srt_path=input_srt, output_script_path=output_script)

    assert output_script.exists()
    # Expect an empty output file
    assert output_script.read_text(encoding="utf-8") == ""

def test_srt_to_script_malformed_blocks(tmp_path: Path):
    """Test conversion with malformed SRT blocks."""
    input_srt = tmp_path / "input_malformed.srt"
    output_script = tmp_path / "output_malformed.txt"
    input_srt.write_text(MALFORMED_SRT_CONTENT, encoding="utf-8")

    srt_to_script(input_srt_path=input_srt, output_script_path=output_script)

    assert output_script.exists()
    # Expect only the text from the valid blocks, malformed ones skipped
    expected_output = "[Speaker A] First line okay.\n\nOkay."
    assert output_script.read_text(encoding="utf-8") == expected_output

def test_srt_to_script_input_not_found(tmp_path: Path):
    """Test that FileNotFoundError is raised if input SRT doesn't exist."""
    non_existent_input = tmp_path / "non_existent.srt"
    output_script = tmp_path / "output.txt"

    with pytest.raises(FileNotFoundError):
        srt_to_script(input_srt_path=non_existent_input, output_script_path=output_script)

    # Ensure output file was not created
    assert not output_script.exists()

def test_srt_to_script_cannot_write_output(tmp_path: Path):
    """Test handling when output cannot be written (e.g., permissions)."""
    input_srt = tmp_path / "input.srt"
    # Create a directory where the output file should be, preventing file creation
    output_dir_as_file = tmp_path / "output.txt"
    output_dir_as_file.mkdir() # Create a directory with the target filename

    input_srt.write_text(SAMPLE_SRT_CONTENT, encoding="utf-8")

    # Expect an IOError or OSError when trying to write the file
    with pytest.raises((IOError, OSError)):
        srt_to_script(input_srt_path=input_srt, output_script_path=output_dir_as_file)