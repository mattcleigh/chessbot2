import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import rootutils
from bulletchess.pgn import BLACK_WON, DRAW_RESULT, UNKNOWN_RESULT, WHITE_WON, PGNFile
from tqdm import tqdm

root = rootutils.setup_root(__file__)

SCHEMA = pa.schema([
    ("moves", pa.string()),
    ("result", pa.int8()),
])

RESULT_MAP = {
    WHITE_WON: 0,
    BLACK_WON: 1,
    DRAW_RESULT: 2,
}


def download_lichess_data(root: Path, start_str: str, end_str: str) -> None:
    root.mkdir(parents=True, exist_ok=True)

    # Convert to datetime objects
    current_dt = datetime.strptime(start_str, "%Y-%m")
    end_dt = datetime.strptime(end_str, "%Y-%m")
    all_dt = []
    while current_dt <= end_dt:
        all_dt.append(current_dt)
        new_month = current_dt.month % 12 + 1
        new_year = current_dt.year + (current_dt.month // 12)
        current_dt = current_dt.replace(year=new_year, month=new_month)

    # Loop until we surpass the end date
    for current_dt in tqdm(all_dt, desc="Downloading Lichess data"):
        date_suffix = current_dt.strftime("%Y-%m")
        file_name = f"lichess_elite_{date_suffix}.zip"
        file_path = root / file_name
        url = f"https://database.nikonoel.fr/{file_name}"

        # Only download if the file doesn't already exist
        if file_path.exists():
            pass
        else:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()

                # Write the whole file at once
                file_path.write_bytes(r.content)
            except Exception as e:
                print(f"Failed. (Reason: {e})")


def process_zip(zip_path: Path, output_dir: Path, worker_index: int) -> int:
    """Worker function to process a single ZIP file end-to-end."""
    # Read ZIP contents directly into memory
    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as pgn_file:
        content = pgn_file.read().decode("utf-8-sig", errors="ignore")

    # Split into multiple games (faster than re) will add delimiter back later
    raw_games = content.split('[Event "')[1:]  # First is empty
    del content  # Free up memory immediately

    # Remove all games that ended due to time forfeit and are too long
    valid_games = [
        '[Event "' + g for g in raw_games if "Time forfeit" not in g and len(g) < 2500
    ]

    # Write to a temp PGN file for parsing with the bulletchess library
    with tempfile.NamedTemporaryFile("w", suffix=".pgn", delete=False) as temp_pgn:
        temp_pgn.write("\n\n".join(valid_games))
        temp_pgn.flush()  # Ensure all data is written to disk before parsing
        del raw_games, valid_games  # Free up memory immediately

    # Parse FENs and Results
    all_results = []
    all_moves = []
    pgn = PGNFile.open(temp_pgn.name)
    pbar = tqdm(desc=f"Parsing {zip_path.name}", position=worker_index, leave=False)
    try:
        while game := pgn.next_game():
            result = game.result

            # Only consider games with known results
            if result == UNKNOWN_RESULT:
                continue

            # Must be a minimum 15 move game
            if len(game.moves) < 15:
                continue

            # Record the result for each position in the game
            all_results.append(RESULT_MAP[result])
            all_moves.append(" ".join(m.uci() for m in game.moves))
            pbar.update(1)
    finally:
        pbar.close()
        Path(temp_pgn.name).unlink()  # Always clean up the temp, even if parsing fails

    # Write out a separate Parquet file for this ZIP
    table = pa.Table.from_arrays(
        [pa.array(all_moves), pa.array(all_results, type=pa.int8())],
        schema=SCHEMA,
    )
    out_filename = f"{Path(zip_path).name.replace('.zip', '')}.parquet"
    out_path = Path(output_dir) / out_filename
    pq.write_table(table, out_path)

    return len(all_moves)


def convert_to_parquet(
    zip_paths: list[Path], output_dir: Path, max_workers: int = 4
) -> None:
    """Main function to distribute the processing workload."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    total_games = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_zip, path, output_dir, (i % max_workers) + 1)
            for i, path in enumerate(zip_paths)
        ]
        pbar = tqdm(
            as_completed(futures),
            total=len(zip_paths),
            desc="Overall Progress",
            unit=" file",
            position=0,
        )
        for future in pbar:
            total_games += future.result()
    print(f"Finished! Total games extracted: {total_games}")


def main():
    data_dir = root / "data"
    date_start = "2024-01"
    date_end = "2025-10"

    # Download the files if they don't exist
    download_lichess_data(data_dir / "zips", date_start, date_end)

    # Split the files into train and test
    all_files = sorted((data_dir / "zips").glob("*"))
    train_files = all_files[:-1]  # Everything before
    test_files = [all_files[-1]]  # Most recent month

    # Save the train and test files in separate folders
    convert_to_parquet(train_files, data_dir / "train", 12)
    convert_to_parquet(test_files, data_dir / "test", 1)


if __name__ == "__main__":
    main()
