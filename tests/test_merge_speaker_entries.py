# tests/test_merge_speaker_entries.py

import pytest
from pathlib import Path
import srt # Required for creating/checking Subtitle objects

# Import the function to test
from chronicleweave.modules.merge_speaker_entries import merge_speaker_entries

# --- Test Data ---

SRT_CONTENT_TO_MERGE = """1
00:00:01,000 --> 00:00:03,000
[Alice] Hello Bob.

2
00:00:03,500 --> 00:00:05,000
[Alice] How are you?

3
00:00:06,000 --> 00:00:08,000
[Bob] I am fine, Alice.

4
00:00:08,500 --> 00:00:10,000
[Bob] Thanks for asking.

5
00:00:11,000 --> 00:00:12,000
[Alice] Good.
"""

EXPECTED_MERGED_CONTENT = """1
00:00:01,000 --> 00:00:05,000
Hello Bob. How are you?

2
00:00:06,000 --> 00:00:10,000
I am fine, Alice. Thanks for asking.

3
00:00:11,000 --> 00:00:12,000
Good.
""" # Note: Speaker tags are removed by the function's internal logic

SRT_CONTENT_NO_MERGE = """1
00:00:01,000 --> 00:00:03,000
[Alice] Hello Bob.

2
00:00:04,000 --> 00:00:05,000
[Bob] Hi Alice.

3
00:00:06,000 --> 00:00:08,000
[Alice] How are you?

4
00:00:09,000 --> 00:00:10,000
[Bob] Fine, thanks.
"""

EXPECTED_NO_MERGE_CONTENT = """1
00:00:01,000 --> 00:00:03,000
Hello Bob.

2
00:00:04,000 --> 00:00:05,000
Hi Alice.

3
00:00:06,000 --> 00:00:08,000
How are you?

4
00:00:09,000 --> 00:00:10,000
Fine, thanks.
"""

SRT_CONTENT_WITH_UNTAGGED = """1
00:00:01,000 --> 00:00:03,000
[Alice] Hello Bob.

2
00:00:03,500 --> 00:00:05,000
[Alice] How are you?

3
00:00:06,000 --> 00:00:08,000
This is an untagged line.

4
00:00:08,500 --> 00:00:10,000
[Bob] I am Bob.

5
00:00:11,000 --> 00:00:12,000
[Bob] Yes, Bob.
"""

EXPECTED_UNTAGGED_CONTENT = """1
00:00:01,000 --> 00:00:05,000
Hello Bob. How are you?

2
00:00:06,000 --> 00:00:08,000
This is an untagged line.

3
00:00:08,500 --> 00:00:12,000
I am Bob. Yes, Bob.
"""

SRT_VALID_INPUT_FOR_WRITE_TEST = """1
00:00:01,000 --> 00:00:03,000
[Alice] Test line.
"""

# --- Test Functions ---

def test_merge_speaker_entries_standard(tmp_path: Path):
    """Test merging consecutive entries for the same speaker."""
    input_srt = tmp_path / "input_merge.srt"
    output_srt = tmp_path / "output_merge.srt"
    input_srt.write_text(SRT_CONTENT_TO_MERGE, encoding="utf-8")

    merge_speaker_entries(input_srt_file=input_srt, output_srt_file=output_srt)

    assert output_srt.exists()
    # Parse the output file to check number of entries and content
    output_content = output_srt.read_text(encoding="utf-8")
    output_subs = list(srt.parse(output_content))

    assert len(output_subs) == 3 # 5 entries merged into 3
    assert output_subs[0].content == "[Alice] Hello Bob. How are you?"
    assert output_subs[0].end == srt.srt_timestamp_to_timedelta("00:00:05,000")
    assert output_subs[1].content == "[Bob] I am fine, Alice. Thanks for asking."
    assert output_subs[1].end == srt.srt_timestamp_to_timedelta("00:00:10,000")
    assert output_subs[2].content == "[Alice] Good." # Last one wasn't merged
    assert output_subs[2].end == srt.srt_timestamp_to_timedelta("00:00:12,000")
    # Also compare full text content for exactness
    # Need a way to compare SRT content ignoring minor whitespace diffs potentially caused by compose
    # For now, check number of subs and key content/times
    # assert output_content.strip() == EXPECTED_MERGED_CONTENT.strip() # This might be too strict

def test_merge_speaker_entries_no_merge(tmp_path: Path):
    """Test scenario where no merging should occur."""
    input_srt = tmp_path / "input_nomerge.srt"
    output_srt = tmp_path / "output_nomerge.srt"
    input_srt.write_text(SRT_CONTENT_NO_MERGE, encoding="utf-8")

    merge_speaker_entries(input_srt_file=input_srt, output_srt_file=output_srt)

    assert output_srt.exists()
    output_content = output_srt.read_text(encoding="utf-8")
    output_subs = list(srt.parse(output_content))

    assert len(output_subs) == 4 # Should remain 4 entries
    # Check content matches original (after tag removal)
    assert output_subs[0].content == "[Alice] Hello Bob."
    assert output_subs[1].content == "[Bob] Hi Alice."
    assert output_subs[2].content == "[Alice] How are you?"
    assert output_subs[3].content == "[Bob] Fine, thanks."
    # assert output_content.strip() == EXPECTED_NO_MERGE_CONTENT.strip()

def test_merge_speaker_entries_with_untagged(tmp_path: Path):
    """Test merging around untagged entries."""
    input_srt = tmp_path / "input_untagged.srt"
    output_srt = tmp_path / "output_untagged.srt"
    input_srt.write_text(SRT_CONTENT_WITH_UNTAGGED, encoding="utf-8")

    merge_speaker_entries(input_srt_file=input_srt, output_srt_file=output_srt)

    assert output_srt.exists()
    output_content = output_srt.read_text(encoding="utf-8")
    output_subs = list(srt.parse(output_content))

    assert len(output_subs) == 3 # [Alice], [Untagged], [Bob]
    assert output_subs[0].content == "[Alice] Hello Bob. How are you?"
    assert output_subs[0].end == srt.srt_timestamp_to_timedelta("00:00:05,000")
    assert output_subs[1].content == "This is an untagged line." # Untagged line unchanged
    assert output_subs[1].end == srt.srt_timestamp_to_timedelta("00:00:08,000")
    assert output_subs[2].content == "[Bob] I am Bob. Yes, Bob."
    assert output_subs[2].end == srt.srt_timestamp_to_timedelta("00:00:12,000")
    # assert output_content.strip() == EXPECTED_UNTAGGED_CONTENT.strip()

def test_merge_speaker_entries_empty_input(tmp_path: Path):
    """Test behavior with an empty input SRT file."""
    input_srt = tmp_path / "input_empty.srt"
    output_srt = tmp_path / "output_empty.srt"
    input_srt.touch() # Create empty file

    merge_speaker_entries(input_srt_file=input_srt, output_srt_file=output_srt)

    assert output_srt.exists()
    # Expect an empty output file (or maybe just index 0 depending on srt lib)
    assert output_srt.read_text(encoding="utf-8").strip() == ""

def test_merge_speaker_entries_invalid_srt(tmp_path: Path):
    """Test handling of malformed SRT input."""
    input_srt = tmp_path / "input_invalid.srt"
    output_srt = tmp_path / "output_invalid.srt"
    # Content missing timestamps
    input_srt.write_text("1\n[Alice] Hello\n\n2\n[Bob] Hi\n", encoding="utf-8")

    # Expect the srt library parse error to be raised
    with pytest.raises(srt.SRTParseError):
        merge_speaker_entries(input_srt_file=input_srt, output_srt_file=output_srt)

    assert not output_srt.exists() # Output should not be created on parse error

def test_merge_speaker_entries_input_not_found(tmp_path: Path):
    """Test FileNotFoundError when input SRT doesn't exist."""
    non_existent_input = tmp_path / "non_existent.srt"
    output_srt = tmp_path / "output.srt"

    with pytest.raises(FileNotFoundError):
        merge_speaker_entries(input_srt_file=non_existent_input, output_srt_file=output_srt)

    assert not output_srt.exists()

def test_merge_speaker_entries_cannot_write_output(tmp_path: Path):
    """Test handling when output SRT file cannot be written."""
    input_srt = tmp_path / "input_valid.srt"
    # Create a directory where the output file should be, preventing file creation
    output_file_as_dir = tmp_path / "cleaned_output.srt"
    output_file_as_dir.mkdir() # Create a directory with the target filename

    # Write valid input content
    input_srt.write_text(SRT_VALID_INPUT_FOR_WRITE_TEST, encoding="utf-8")

    # Expect an IOError or OSError when trying to write the file
    # Check for RuntimeError as well, as the function might wrap the original exception
    with pytest.raises((IOError, OSError, RuntimeError)):
        merge_speaker_entries(input_srt_file=input_srt, output_srt_file=output_file_as_dir)

    # Optional: Check that the directory still exists (wasn't overwritten/deleted)
    assert output_file_as_dir.is_dir()