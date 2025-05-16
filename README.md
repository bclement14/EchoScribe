# 🎲 ChronicleWeave - From D&D Sessions to Living Stories

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> Capture the magic of your tabletop adventures — from chaotic voices to polished, readable narratives.

---

## 🧙‍♂️ What is ChronicleWeave?

**ChronicleWeave** is a modular Python pipeline designed to transform **multi-speaker audio recordings** (like D&D sessions) into structured, readable scripts.

It automates the process from individual audio tracks per speaker to a final text document, handling:

1.  **Audio Chunking:** Intelligently splits long recordings based on silence, optimized for speaker-specific tracks.
2.  **Transcription:** Uses the powerful **[WhisperX](https://github.com/m-bain/whisperX)** model via Docker for accurate speech-to-text.
3.  **Correction & Formatting:** Cleans and corrects transcription timestamp anomalies for better accuracy.
4.  **Dialogue Assembly:** Merges transcriptions from different chunks and speakers into a coherent, timestamped SRT file.
5.  **Script Generation:** Creates a final plain text script suitable for reading or further processing.

### 🚀 **Incoming:** Module for LLM-based summarization or narrative generation.

Read the full story behind ChronicleWeave in the **🌱 Context and Origin** section.

---

## 📦 Key Features

- 🎧 **Silence-based Audio Chunking** optimized for multi-track recordings.
- 🐳 **Dockerized WhisperX Integration** for robust, GPU-accelerated transcription.
- 🛠️ **Advanced Timestamp Correction** using statistical analysis.
- ⏱️ **SRT Subtitle Generation** with speaker labels.
- 💬 **Speaker Turn Merging** for cleaner SRT output.
- 📝 **Plain Text Script Generation**.
- 🧹 **Low RAM Mode** option for audio processing on systems with limited memory.
- 🗂️ **Organized Folder Structure** per session.
- 🔧 **Configurable Pipeline Steps**.

---

## ⚙️ System Requirements

1.  **Python:** >= 3.8
2.  **Docker:** Required for running WhisperX transcription. Docker Engine needs to be installed and running. GPU access configured within Docker is highly recommended for performance. [Install Docker](https://docs.docker.com/engine/install/)
3.  **FFmpeg:** Required by the `pydub` library for audio manipulation.
    - Install via your system's package manager (e.g., `sudo apt install ffmpeg`, `brew install ffmpeg`) or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your system's PATH.
    - Verify with: `ffmpeg -version`

---

## 🐳 WhisperX Transcription via Docker

ChronicleWeave leverages the powerful **[WhisperX](https://github.com/m-bain/whisperX)** library for transcription and timestamp alignment. To ensure a consistent and reliable environment, especially regarding specific CUDA and cuDNN dependencies that can sometimes cause issues when installed directly, ChronicleWeave runs WhisperX within a **Docker container**.

**Why Docker?**

- **Dependency Management:** Encapsulates the specific versions of WhisperX, PyTorch, CUDA, cuDNN, and other dependencies known to work together reliably. This avoids common environment setup problems reported by users online.
- **Reproducibility:** Ensures the transcription step runs the same way regardless of the host machine's specific Python or CUDA setup.
- **Ease of Use:** The pipeline handles running the container, mounting volumes, and passing arguments automatically when `run_whisperx=True`.

**Setup:**

1.  **Install Docker Engine:** Follow instructions at [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/).
2.  **Configure GPU Access (Recommended):** For significantly faster transcription, ensure Docker can access your NVIDIA GPU. See [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
3.  **Build the Image:** From the project root directory (containing the `Dockerfile`), run:
    ```bash
    docker build -t chronicleWeave-whisperx .
    ```
    _(This uses the included Dockerfile based on official NVIDIA CUDA images and installs the necessary WhisperX version)._

Once built, the `run_pipeline` function will use the `chronicleWeave-whisperx` image automatically for Step 2.

---

## 🚀 Quickstart & Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/bclement14/chronicleWeave.git
cd chronicleWeave
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv .venv
# Activate it:
source .venv/bin/activate  # On Linux/macOS
# .\.venv\Scripts\activate  # On Windows PowerShell
# .\venv\Scripts\activate.bat # On Windows Cmd
```

### 3. Install Dependencies

Install the package in editable mode (`-e`) along with development dependencies (like `pytest`):

```bash
pip install -e ".[dev]"
```

If you only want to install the package for usage (without test dependencies):

```bash
pip install .
```

### 4. Build the WhisperX Docker Image

Build the Docker image required by the pipeline once:

```bash
docker build -t chronicleWeave-whisperx .
```

_(Ensure the Dockerfile is present in the project root)_

### 5. Prepare Your Session Folder

Create a base directory for your session and place the individual speaker audio tracks (e.g., FLAC files from Craig) inside a subfolder named `tracks` (or your configured input folder name):

```
your_session_name/
└── tracks/
    ├── speaker1.flac
    ├── speaker2.flac
    └── speaker3.flac
```

### 6. Run the Pipeline (Example)

Create a Python script (e.g., `process_session.py`) or use an interactive Python session (like IPython/Jupyter):

```python
from chronicleWeave.pipeline import run_pipeline
from pathlib import Path

# Define the path to your session directory
session_path = Path("./your_session_name") # Use relative or absolute path

# Basic run (using most defaults)
run_pipeline(base_path=str(session_path))

# --- OR ---

# Run with specific options
run_pipeline(
    base_path=str(session_path),
    # input_audio_folderName="craig_audio", # Example: If your folder isn't 'tracks'
    diarize=False, # Example: Disable diarization if not needed
    use_low_ram=True,
    log_level="INFO", # Set to "DEBUG" for more detail
    # Run only specific steps (e.g., chunking and transcription)
    # steps_to_run=[1, 2]
    # steps_to_run=slice(1, 4) # Steps 1, 2, 3
)

print(f"Pipeline processing finished for {session_path.name}!")
```

---

## 🧹 Pipeline Steps Overview

The `run_pipeline` function executes the following steps (configurable via `steps_to_run`):

| Step | Module (`chronicleWeave.modules.*`) | Action                       | Output Location (Default)        |
| :--- | :---------------------------------- | :--------------------------- | :------------------------------- |
| 1    | `audio_chunker`                     | Chunk audio based on silence | `chunked_tracks/`                |
| 2    | `pipeline` (internal Docker call)   | Transcribe chunks (WhisperX) | `wx_output/`                     |
| 3    | `whisperx_corrector_core`           | Correct JSON timestamps      | `json_files/`                    |
| 4    | `convert_json_to_srt`               | Convert JSON to SRT          | `srt_files/`                     |
| 5    | `merge_srt_by_chunk`                | Merge chunk SRTs w/ offset   | `final_outputs/merged_*.srt`     |
| 6    | `merge_speaker_entries`             | Merge consecutive speakers   | `final_outputs/cleaned_*.srt`    |
| 7    | `convert_srt_to_script`             | Create plain text script     | `final_outputs/final_script.txt` |

Intermediate files are stored in subdirectories within your `base_path`. Final key outputs land in the `final_outputs` directory by default.

---

## 🐳 Manual Docker Command (Advanced)

If you need to run the WhisperX transcription step manually outside the pipeline:

1.  Ensure your chunked audio files (e.g., `.flac`) are in a subdirectory (e.g., `chunked_tracks`).
2.  Build the image: `docker build -t chronicleWeave-whisperx .`
3.  Run the container:

    ```bash
    # Example using 'find' (safer for many files/special chars) on Linux/macOS:
    docker run --gpus all -it --rm -v "$(pwd):/app" --ipc=host \
      chronicleWeave-whisperx \
      whisperx $(find chunked_tracks -name '*.flac' -printf '%p ') \
      --model large-v3 --language fr \
      --output_dir wx_output --output_format json \
      # Add --diarize --hf_token YOUR_TOKEN if needed
    ```

    **OR**

    ```bash
    # Example using shell wildcard (CAUTION: May fail with many files or special characters):
    docker run --gpus all -it --rm -v "$(pwd):/app" --ipc=host \
      chronicleWeave-whisperx \
      whisperx chunked_tracks/*.flac \
      --model large-v3 --language fr \
      --output_dir wx_output --output_format json \
      # Add --diarize --hf_token YOUR_TOKEN if needed
    ```

Place the resulting JSON files from `wx_output/` into the expected directory for the pipeline's Step 3 (default: same `wx_output/` name relative to your `base_path`).

---

## 📂 Example Final Folder Structure

```
your_session_name/
├── tracks/                     # Input speaker audio files
├── chunked_tracks/             # Output from Step 1
├── wx_output/                  # Output from Step 2 (WhisperX JSON)
├── json_files/                 # Output from Step 3 (Corrected JSON)
├── srt_files/                  # Output from Step 4 (Chunked SRTs)
└── final_outputs/              # Final aggregated outputs
    ├── merged_audio.flac       # Merged audio (from Step 1)
    ├── cut_points.txt          # Silence cut points (from Step 1)
    ├── merged_transcript.srt   # Output from Step 5
    ├── cleaned_transcript.srt  # Output from Step 6
    └── final_script.txt        # Output from Step 7
```

---

## 🧪 Testing

To run the unit tests:

1.  Ensure development dependencies are installed: `pip install -e ".[dev]"`
2.  Run pytest from the project root directory:
    ```bash
    pytest
    ```

---

## 🌱 Context and Origin

ChronicleWeave started as a personal quest with a perhaps ambitious goal: I wanted to automatically transcribe my D&D group's recorded sessions and use LLMs to generate summaries or session-based stories, dreaming of potentially weaving those adventures into a novel someday.

Like many D&D groups we used Discord for our sessions and we used the **[Craig bot](https://craig.chat/)** 🐻 to record the sessions. It provides separate audio tracks for each speaker (multi-track recording) – a feature that proved crucial later on. My first attempts, however, involved manually chunking the audio using tools like Audacity and trying the original [Whisper](https://github.com/openai/whisper) transcription model. The results weren't ideal, especially with the long silences present in individual speaker tracks (often, these silences, where other speakers were active, constituted more time than actual speech within a given chunk).

Checking for better models, I discovered **[WhisperX](https://github.com/m-bain/whisperX)**. My initial experiments focused on simplifying the input by using a _merged_ audio track and leveraging WhisperX's built-in diarization (`--diarize`) to separate speakers. Unfortunately, even after several rounds of parameter fine-tuning, it often failed to identify speakers correctly, which was a crucial requirement for the automated pipeline I envisioned.

This led me back to using the individual speaker tracks. Processing these separately gave much clearer transcriptions, as WhisperX appeared to handle the extensive silences within these tracks more effectively than the original Whisper model had in my earlier tests. However, a new challenge arose: due to these same long silences, WhisperX's word-level timestamp alignment was sometimes inaccurate. This, in turn, led to poor temporal synchronization when attempting to merge the dialogues from different speaker tracks into a cohesive script, which led to a misleading understanding of the temporal continuum by the LLM and poor restitution of the actual events.

To address this, I developed a custom script to process WhisperX's JSON output, detect timestamp inconsistencies (like words with improbable durations or gaps), and correct them using a home-made heuristic. This correction step became a cornerstone. My workflow evolved into a pipeline of several independent Python scripts: one to analyze the combined audio for silence and then chunk the individual speaker tracks using common cut points (ensuring temporal consistency), another for the WhisperX JSON correction, one more to reassemble the transcribed chunks into a full script, and finally, manually prompting Large Language Models (LLMs) to generate summaries or even pre-novelized versions of the sessions.

While functional for my own use, this multi-script, manual process was cumbersome. I wanted a fully automated pipeline that I could run right after each session, delivering the LLM outputs automatically.

Realizing others might have similar goals – whether for creating campaign summaries, session recaps for players, or other creative uses – the idea formed to transform these scripts not only into a personal automated pipeline but into a more robust, automated, and shareable library.

That's where this version of ChronicleWeave comes from. It has been significantly refactored and improved from those initial scripts into a structured, installable Python library (`chronicleWeave`) with considerable assistance from Large Language Models (such as Gemini 2.5 Pro, GPT-4o, and Claude 3.5 Sonnet\*). The goal of this LLM-guided refactoring was to consolidate the steps, enhance robustness, add proper error handling, improve maintainability through modern Python practices, incorporate unit testing, increase usability, and prepare the code for open-sourcing on GitHub. The development involved an iterative process of code generation, detailed review, and refinement, often guided by the LLMs.

---

## 💡 Future Directions

### Next building block

- [ ] **LLM Integration:** Summarization, narrative generation, character dialogue extraction through LLM API.
- [ ] **Pipeline Running Example:** Find audio sample to allow pipeline running for testing.

### Possible evolutions

- [ ] **Direct Discord Bot Integration:** Streamline the process from recording to script.
- [ ] **Alternative ASR Models:** Integration options beyond WhisperX.
- [ ] **Configuration Files:** Allow pipeline configuration via YAML/TOML files.

---

## 🧑‍💻 License

This project is open-source under the **GNU GENERAL PUBLIC LICENSE v3.0**. See the `LICENSE` file for details.

<!-- ---

# 📣 Join the Adventure!

Contributions, feedback, and feature requests are welcome! Help make ChronicleWeave the ultimate tool for chronicling tabletop stories. Create an issue or submit a pull request. ⚔️📜 -->
