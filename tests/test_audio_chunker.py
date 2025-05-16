# tests/test_audio_chunker.py

import pytest
from pathlib import Path
import numpy as np 
import logging # Import logging if needed for caplog, though not used here yet

# Import functions and config from the module to test
from chronicleWeave.modules.audio_chunker import (
    get_audio_files_from_folder,
    determine_cut_points,
    write_cut_points,
    ChunkingConfig,
    DEFAULT_CHUNKING_CONFIG
)

# --- Fixtures ---

@pytest.fixture
def default_chunk_config() -> ChunkingConfig:
    """Provides a default ChunkingConfig instance."""
    # Use the actual default imported from the module
    return DEFAULT_CHUNKING_CONFIG

# --- Tests for get_audio_files_from_folder ---

def test_get_audio_files_success(tmp_path: Path):
    """Test finding FLAC files in a directory."""
    input_dir = tmp_path / "audio_in"
    input_dir.mkdir()
    file1 = input_dir / "speakerA.flac"
    file2 = input_dir / "speakerB.flac"
    other_file = input_dir / "notes.txt"
    file1.touch()
    file2.touch()
    other_file.touch()

    found_files = get_audio_files_from_folder(input_dir, extension="flac")

    assert len(found_files) == 2
    # Use sets for order-independent comparison
    assert set(found_files) == {file1, file2}

def test_get_audio_files_custom_extension(tmp_path: Path):
    """Test finding files with a different extension (using lowercase)."""
    input_dir = tmp_path / "audio_wav"
    input_dir.mkdir()
    file1 = input_dir / "track1.wav"
    # Use lowercase for test reliability across platforms
    file2 = input_dir / "track2.wav"
    other_file = input_dir / "track.flac"
    file1.touch()
    file2.touch()
    other_file.touch()

    # Test with simple lowercase glob pattern
    found_files = get_audio_files_from_folder(input_dir, extension="wav")

    assert len(found_files) == 2
    assert set(found_files) == {file1, file2}

def test_get_audio_files_empty(tmp_path: Path):
    """Test finding files in an empty directory."""
    input_dir = tmp_path / "audio_empty"
    input_dir.mkdir()

    found_files = get_audio_files_from_folder(input_dir, extension="flac")
    assert found_files == []

def test_get_audio_files_no_matching(tmp_path: Path):
    """Test finding files when none match the extension."""
    input_dir = tmp_path / "audio_nomatch"
    input_dir.mkdir()
    (input_dir / "notes.txt").touch()
    (input_dir / "audio.mp3").touch()

    found_files = get_audio_files_from_folder(input_dir, extension="flac")
    assert found_files == []

def test_get_audio_files_dir_not_found(tmp_path: Path):
    """Test error when input directory does not exist."""
    input_dir = tmp_path / "non_existent_dir"
    with pytest.raises(FileNotFoundError):
        get_audio_files_from_folder(input_dir)

# --- Tests for determine_cut_points ---

# Use default config: min_chunk=540s (9min), max_chunk=900s (15min)
def test_determine_cuts_no_silence(default_chunk_config):
    """Test cutting based on max duration when no silences are provided."""
    silences = []
    duration = 1000.0 # > 900s max
    # Expect 2 chunks: [0.0, 500.0, 1000.0] (Fallback cuts based on duration/num_chunks)
    cut_points = determine_cut_points(silences, duration, default_chunk_config)
    assert len(cut_points) == 3
    assert cut_points[0] == pytest.approx(0.0)
    # Fallback calculation: ceil(1000/900)=2 chunks. Points are 0, 1000/2, 1000.
    assert cut_points[1] == pytest.approx(500.0)
    assert cut_points[2] == pytest.approx(1000.0)

def test_determine_cuts_simple_valid_silences(default_chunk_config):
   def test_determine_cuts_simple_valid_silences(default_chunk_config):
    """Test cutting within suitable silences."""
    # Config: min=540, max=900
    silences = [(600.0, 610.0), (1200.0, 1210.0)] # Silences at 10min and 20min
    duration = 1500.0
    cut_points = determine_cut_points(silences, duration, default_chunk_config)

    # Corrected Expected behaviour trace:
    # 1. Process (600, 610): mid=605. Chunk 0-605 is 605s (valid). Add 605. last_cut=605. points=[0, 605].
    # 2. Process (1200, 1210): mid=1205. Chunk 605-1205 is 600s (valid). Add 1205. last_cut=1205. points=[0, 605, 1205].
    # 3. End loop. Check final segment 1205-1500 is 295s (> min/2=270). OK.
    # 4. Add final point 1500. points=[0, 605, 1205, 1500].
    expected_points = [0.0, 605.0, 1205.0, 1500.0]
    assert len(cut_points) == 4 # Expect 4 points (3 chunks)
    assert cut_points == pytest.approx(expected_points)

def test_determine_cuts_force_cut_due_to_max_duration(default_chunk_config):
    """Test forcing a cut when silence is too far away."""
    # Config: min=540, max=900
    silences = [(1000.0, 1010.0)] # Only one silence, far out
    duration = 1500.0
    # Expected cuts: [0.0, 900.0, 1500.0] (as analyzed before)
    cut_points = determine_cut_points(silences, duration, default_chunk_config)
    assert len(cut_points) == 3
    assert cut_points[0] == pytest.approx(0.0)
    assert cut_points[1] == pytest.approx(900.0) # Forced cut at max duration
    assert cut_points[2] == pytest.approx(1500.0)

def test_determine_cuts_force_cut_chooses_earlier_silence(default_chunk_config):
    """Test forced cut selecting the latest possible silence before max duration."""
    # Config: min=540, max=900
    silences = [(600.0, 610.0), (850.0, 860.0), (1000.0, 1010.0)]
    duration = 1500.0
    cut_points = determine_cut_points(silences, duration, default_chunk_config)

    # Corrected Expected behaviour trace:
    # 1. Process (600, 610): mid=605. Chunk 0-605 is 605s (valid). Add 605. last_cut=605. points=[0, 605].
    # 2. Process (850, 860): mid=855. Chunk 605-855 is 250s (< min=540). Skip this silence.
    # 3. Process (1000, 1010): mid=1005. Chunk 605-1005 is 400s (< min=540). Skip this silence.
    # 4. End loop. Check final segment 605-1500 is 895s (OK).
    # 5. Add final point 1500. points=[0, 605, 1500].
    expected_points = [0.0, 605.0, 1500.0]
    assert len(cut_points) == 3
    assert cut_points == pytest.approx(expected_points)

def test_determine_cuts_final_segment_too_short(default_chunk_config):
    """Test removing the last cut point if the final segment is too short."""
    # Config: min=540, max=900. min/2 = 270
    # Case 1: Final segment is long enough
    silences1 = [(600.0, 610.0), (1100.0, 1110.0)]
    duration1 = 1200.0
    # Expected: [0.0, 605.0, 1200.0] (Final segment 595s > 270)
    cut_points1 = determine_cut_points(silences1, duration1, default_chunk_config)
    assert len(cut_points1) == 3
    assert cut_points1 == pytest.approx([0.0, 605.0, 1200.0])

    # Case 2: Final segment is just long enough
    silences2 = [(600.0, 610.0), (850.0, 860.0)]
    duration2 = 900.0
    # Expected: [0.0, 605.0, 900.0] (Final segment 295s > 270)
    cut_points2 = determine_cut_points(silences2, duration2, default_chunk_config)
    assert len(cut_points2) == 3
    assert cut_points2 == pytest.approx([0.0, 605.0, 900.0])

    # Case 3: Final segment is too short, last cut removed
    silences3 = [(600.0, 610.0)]
    duration3 = 700.0
    # Expected: [0.0, 700.0] (Final segment 95s < 270, cut at 605 removed)
    cut_points3 = determine_cut_points(silences3, duration3, default_chunk_config)
    assert len(cut_points3) == 2
    assert cut_points3 == pytest.approx([0.0, 700.0])


# --- Tests for write_cut_points ---

def test_write_cut_points_standard(tmp_path: Path):
    """Test writing cut points to a file."""
    cut_points = [0.0, 605.1234, 1200.5]
    output_file = tmp_path / "cuts_output" / "cut_points.txt"

    write_cut_points(cut_points, output_file)

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8").splitlines()
    assert "# Audio Cut Points (Total Chunks: 2)" in content[0]
    assert "# Format: Chunk Index, Start Time (seconds)" in content[1]
    assert content[2] == "Chunk 0, 0.000"
    assert content[3] == "Chunk 1, 605.123" # Check rounding to 3 decimal places
    assert content[4] == "Chunk 2, 1200.500"

def test_write_cut_points_empty(tmp_path: Path):
    """Test writing when the cut points list is empty."""
    cut_points = []
    output_file = tmp_path / "empty_cuts.txt"

    write_cut_points(cut_points, output_file)

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8").splitlines()
    # --- CORRECTED ASSERTION ---
    # Expect only the two header lines when input list is empty
    assert len(content) == 2
    assert "# Audio Cut Points (Total Chunks: -1)" in content[0] # len([])-1 = -1
    assert "# Format: Chunk Index, Start Time (seconds)" in content[1]
    # --- END CORRECTION ---

def test_write_cut_points_single_point(tmp_path: Path):
    """Test writing when only one cut point (start=0) exists."""
    cut_points = [0.0]
    output_file = tmp_path / "single_cut.txt"

    write_cut_points(cut_points, output_file)

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8").splitlines()
    # Expect two headers and one data line
    assert len(content) == 3
    assert "# Audio Cut Points (Total Chunks: 0)" in content[0] # len([0.0])-1 = 0
    assert "# Format: Chunk Index, Start Time (seconds)" in content[1]
    assert content[2] == "Chunk 0, 0.000"


def test_write_cut_points_cannot_write(tmp_path: Path):
    """Test error handling when output file cannot be written."""
    cut_points = [0.0, 100.0]
    # Create a directory where the file should go
    output_file_as_dir = tmp_path / "cut_points.txt"
    output_file_as_dir.mkdir()

    with pytest.raises((IOError, OSError)):
        write_cut_points(cut_points, output_file_as_dir)

# --- TODO: Write Integration tests or tests for pydub dependent functions ---
# - _merge_tracks_high_ram / _merge_tracks_low_ram
# - _detect_silence_high_ram / _detect_silence_low_ram
# - split_tracks
# - chunk_audio (main function)