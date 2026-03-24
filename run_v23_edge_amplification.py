from __future__ import annotations

from research import run_v23_edge_amplification


def main() -> None:
    artifacts = run_v23_edge_amplification()
    print("V2.3 edge amplification run completed.")
    print(artifacts.summary.round(6).to_string(index=False))
    print(f"Outputs written to: {artifacts.output_dir.resolve()}")


if __name__ == "__main__":
    main()
