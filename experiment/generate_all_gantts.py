from pathlib import Path
import sys

# Ensure project root is on sys.path so utils can be imported from experiment/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.gantt_diagrams import generate_comparison_from_json


def main():
    results_dir = Path(__file__).resolve().parent / "results"
    base_output_dir = Path(__file__).resolve().parent / "gantt_diagrams"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    total_processed = 0
    total_generated = 0

    json_files = list(results_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files under {results_dir}")

    for json_file in json_files:
        # Mirror directory structure relative to results_dir
        rel_path = json_file.relative_to(results_dir)
        mirrored_dir = base_output_dir / rel_path.parent
        mirrored_dir.mkdir(parents=True, exist_ok=True)

        # Use the JSON stem as the base filename and save as PNG
        output_filename = f"{json_file.stem}.png"

        print(f"Processing {rel_path}...")
        total_processed += 1

        try:
            generate_comparison_from_json(
                filepath=str(json_file),
                output_dir=str(mirrored_dir),
                filename=output_filename,
                mip_key='milp'
            )
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

        # Check output file existence to count successful generations
        out_path = mirrored_dir / output_filename
        if out_path.exists():
            total_generated += 1
        else:
            # The plotting function may skip generating for some files; report that
            print(f"No diagram was created for {json_file} (may have been skipped).")

    print(f"Done. Processed {total_processed} files. Generated {total_generated} diagrams.")


if __name__ == "__main__":
    main()
