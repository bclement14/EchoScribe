# echoscribe/modules/convert_srt_to_script.py

import re
import os

def srt_to_script(input_srt_path: str, output_script_path: str) -> None:
    """
    Convert an SRT file into a plain text script by removing index and timestamp lines.

    Args:
        input_srt_path (str): Path to the input SRT file.
        output_script_path (str): Path to save the resulting plain text script.
    """
    print(f"Converting SRT '{input_srt_path}' into plain text script...")

    if not os.path.isfile(input_srt_path):
        raise FileNotFoundError(f"Input SRT file not found: {input_srt_path}")

    with open(input_srt_path, "r", encoding="utf-8") as infile:
        content = infile.read()

    # Split SRT into blocks based on blank lines
    blocks = re.split(r'\n\s*\n', content.strip())

    output_lines = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) >= 3:
            # Text lines are after index and timestamp
            text = " ".join(lines[2:]).strip()
            output_lines.append(text)

    os.makedirs(os.path.dirname(output_script_path), exist_ok=True)
    with open(output_script_path, "w", encoding="utf-8") as outfile:
        outfile.write("\n\n".join(output_lines))

    print(f"Plain text script saved to '{output_script_path}'.")
