import re
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import rootutils

root = rootutils.setup_root(__file__)

# 1. Look for a line starting with "1."
# 2. Capture everything until we see one of the 4 valid PGN endings
# 3. Use a capturing group for the moves and a second one for the result
GAME_RE = re.compile(r"\n(1\..*?)\s(1-0|0-1|1/2-1/2|\*)(?=\n|\Z)", re.DOTALL)
SCHEMA = pa.schema([
    ("moves", pa.string()),  # Mainline moves in UCI format, e.g. "e2e4 e7e5 g1f3"
    ("result", pa.int8()),  # 0 = White win, 1 = Black win, 2 = Draw
])

RESULT_MAP = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}


def process_files(zip_paths, output_path):
    writer = pq.ParquetWriter(
        output_path,
        SCHEMA,
        compression="zstd",
        compression_level=9,
        data_page_size=1024 * 1024,
    )
    total = 0
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as pgn_file:
            content = pgn_file.read().decode("utf-8", errors="ignore")
            games = GAME_RE.findall(content)
            batch_moves = []
            batch_results = []
            for moves, result in games:
                mapped = RESULT_MAP.get(result)
                if mapped is None:
                    continue  # Skip ambiguous / unknown results
                tokens = [t for t in moves.split() if not re.match(r"\d+\.+", t)]
                batch_moves.append(" ".join(tokens))
                batch_results.append(mapped)
            table = pa.Table.from_arrays(
                [pa.array(batch_moves), pa.array(batch_results, type=pa.int8())],
                schema=SCHEMA,
            )
            writer.write_table(table)
            total += len(batch_moves)
            print(f"Added {len(batch_moves):,} games from {zip_path.name}")
    writer.close()
    print(f"Wrote {total:,} games to {output_path}")


def main():
    data_dir = root / "data"
    train_files = [
        "lichess_elite_2025-01.zip",
        "lichess_elite_2025-02.zip",
        "lichess_elite_2025-03.zip",
        "lichess_elite_2025-04.zip",
        "lichess_elite_2025-05.zip",
        "lichess_elite_2025-06.zip",
        "lichess_elite_2025-07.zip",
        "lichess_elite_2025-08.zip",
        "lichess_elite_2025-09.zip",
        "lichess_elite_2025-10.zip",
    ]
    test_files = [
        "lichess_elite_2025-11.zip",
    ]

    process_files([data_dir / x for x in train_files], data_dir / "train.parquet")
    process_files([data_dir / x for x in test_files], data_dir / "test.parquet")


if __name__ == "__main__":
    main()
