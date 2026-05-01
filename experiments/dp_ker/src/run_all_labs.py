from __future__ import annotations

from convert_hf_dataset import main as convert_main
from inspect_dataset import main as inspect_main
from run_ablation import main as ablation_main
from run_experiment import main as experiment_main
from run_privacy_budget import main as privacy_main


def main() -> None:
    inspect_main()
    convert_main()
    experiment_main()
    ablation_main()
    privacy_main()


if __name__ == "__main__":
    main()

