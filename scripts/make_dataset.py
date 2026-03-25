import re
import zipfile
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import rootutils
from tqdm import tqdm

root = rootutils.setup_root(__file__)

# 1. Look for a line starting with "1."
# 2. Capture everything until we see one of the 4 valid PGN endings
# 3. Use a capturing group for the moves and a second one for the result
RESULT_MAP = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}
GAME_RE = re.compile(r"\n(1\..*?)\s(1-0|0-1|1/2-1/2|\*)(?=\n|\Z)", re.DOTALL)
CLEAN_RE = re.compile(r"\s*\d+\.(?:\.\.)?\s*|\s+")
SCHEMA = pa.schema([
    ("moves", pa.string()),  # Mainline moves in UCI format, e.g. "e2e4 e7e5 g1f3"
    ("result", pa.int8()),  # 0 = White win, 1 = Black win, 2 = Draw
])


def download_lichess_data(root: Path, start_str: str, end_str: str) -> None:
    root.mkdir(parents=True, exist_ok=True)

    # Convert to datetime objects
    current_dt = datetime.strptime(start_str, "%Y-%m")
    end_dt = datetime.strptime(end_str, "%Y-%m")

    # Loop until we surpass the end date
    while current_dt <= end_dt:
        date_suffix = current_dt.strftime("%Y-%m")
        file_name = f"lichess_elite_{date_suffix}.zip"
        file_path = root / file_name
        url = f"https://database.nikonoel.fr/{file_name}"

        # Only download if the file doesn't already exist
        if file_path.exists():
            print(f"Skipping {file_name} (already exists)")
        else:
            print(f"Downloading {file_name}...", end=" ", flush=True)
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()

                # Write the whole file at once
                file_path.write_bytes(r.content)
                print("Done.")
            except Exception as e:
                print(f"Failed. (Reason: {e})")

        # Logic to increment the month by 1
        new_month = current_dt.month % 12 + 1
        new_year = current_dt.year + (current_dt.month // 12)
        current_dt = current_dt.replace(year=new_year, month=new_month)


def combine_to_parquet(output_path: Path, zip_paths: list[Path]) -> None:
    writer = pq.ParquetWriter(output_path, SCHEMA)
    total = 0
    for zip_path in tqdm(zip_paths, desc="Combining ZIPs", unit="file"):
        with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as pgn_file:
            content = pgn_file.read().decode("utf-8", errors="ignore")
        # content = content.replace("\r\n", "\n")  # Normalize newlines
        games = GAME_RE.findall(content)

        batch_moves = []
        batch_results = []
        for moves, result in tqdm(
            games,
            desc=f"Processing {zip_path.name}",
            unit="game",
            leave=False,
        ):
            mapped = RESULT_MAP.get(result)
            if mapped is None:
                continue  # Skip ambiguous / unknown results
            cleaned = CLEAN_RE.sub(" ", moves).strip()
            if len(cleaned) < 10:
                continue  # Skip very short games
            batch_moves.append(cleaned)
            batch_results.append(mapped)
        table = pa.Table.from_arrays(
            [pa.array(batch_moves), pa.array(batch_results, type=pa.int8())],
            schema=SCHEMA,
        )
        writer.write_table(table)
        total += len(batch_moves)
        tqdm.write(f"Added {len(batch_moves):,} games from {zip_path.name}")
    writer.close()
    print(f"Wrote {total:,} games to {output_path}")


def main():
    data_dir = root / "data"
    date_start = "2025-01"
    date_end = "2025-10"

    # Download the files if they don't exist
    download_lichess_data(data_dir, date_start, date_end)

    # Combine into a single Parquet file
    all_files = sorted(data_dir.glob("lichess_elite_2025*.zip"))
    train_files = all_files[:-1]  # Everything before
    test_files = [all_files[-1]]  # Most recent month
    combine_to_parquet(data_dir / "train.parquet", train_files)
    combine_to_parquet(data_dir / "test.parquet", test_files)


if __name__ == "__main__":
    main()
