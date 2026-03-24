from __future__ import annotations

from research import run_v22_candidate_hardening


def main() -> None:
    artifacts = run_v22_candidate_hardening()
    print("V2.2 candidate hardening run completed.")
    print(artifacts.summary.round(6).to_string(index=False))
    print(f"Outputs written to: {artifacts.output_dir.resolve()}")


if __name__ == "__main__":
    main()
