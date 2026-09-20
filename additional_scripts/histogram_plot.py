from __future__ import annotations

from pathlib import Path
import re
import shutil

# Assuming UpdateFrequencyPictureGenerator is imported from your module:
from supervisor.visualize import UpdateFrequencyPictureGenerator

def get_newest_folder(logs_dir: Path) -> Path:
    folders = [p for p in logs_dir.iterdir() if p.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No subdirectories found in {logs_dir}")
    return max(folders, key=lambda p: p.stat().st_mtime)


def parse_continuation(stats_file: Path) -> Path | None:
    if not stats_file.exists():
        return None

    pattern = re.compile(r"CONTINUATION FROM SESSION:\s*(.+)", re.IGNORECASE)
    with stats_file.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line.strip())
            if match:
                # Extracts the raw path string (e.g. "logs\2026-09-16_00-20")
                raw_path = match.group(1).strip()
                return Path(raw_path)
    return None


def main():
    logs_root = Path("logs")
    if not logs_root.exists():
        raise FileNotFoundError(f"'{logs_root}' directory does not exist.")

    # 1. Locate the newest session directory
    newest_dir = get_newest_folder(logs_root)
    print(f"Newest session: {newest_dir.name}")

    target_timestamps_path = newest_dir / "all_timestamps.txt"
    src_timestamps_path = newest_dir / "timestamps.txt"

    if not src_timestamps_path.exists():
        raise FileNotFoundError(f"Missing '{src_timestamps_path}'")

    # 2. Initialize all_timestamps.txt as a copy of the newest timestamps.txt
    shutil.copyfile(src_timestamps_path, target_timestamps_path)

    # 3. Traverse backward through continuation chains
    current_dir = newest_dir
    visited_dirs = {current_dir.resolve()}

    with target_timestamps_path.open("a", encoding="utf-8") as out_file:
        while True:
            stats_file = current_dir / "execution_statistics.txt"
            continued_rel_path = parse_continuation(stats_file)

            if not continued_rel_path:
                break

            # Handle both relative paths like 'logs/folder' or direct subfolder names
            if (logs_root.parent / continued_rel_path).exists():
                next_dir = logs_root.parent / continued_rel_path
            elif (logs_root / continued_rel_path.name).exists():
                next_dir = logs_root / continued_rel_path.name
            else:
                print(f"Warning: Continuation folder '{continued_rel_path}' not found. Stopping traversal.")
                break

            next_resolved = next_dir.resolve()
            if next_resolved in visited_dirs:
                print(f"Loop detected at {next_dir}. Stopping traversal.")
                break
            visited_dirs.add(next_resolved)

            prev_timestamps = next_dir / "timestamps.txt"
            if prev_timestamps.exists():
                print(f"Appending timestamps from: {next_dir.name}")
                content = prev_timestamps.read_text(encoding="utf-8")
                # Ensure spacing between appended segments
                if not content.startswith("\n") and out_file.tell() > 0:
                    out_file.write("\n")
                out_file.write(content)
            else:
                print(f"Warning: '{prev_timestamps}' does not exist.")

            current_dir = next_dir

    # 4. Generate the visualization
    pic = UpdateFrequencyPictureGenerator(
        logs_root=str(logs_root),
        session_dir=newest_dir.name,
        timestamp_file="all_timestamps.txt",
        bin_size_sec=300.0,
    )
    pic.run()


if __name__ == "__main__":
    main()