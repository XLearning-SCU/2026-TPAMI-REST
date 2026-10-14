"""REST evaluation entrypoint for composed image retrieval."""
import runpy
import sys
import os
from pathlib import Path


def pop_dataset_arg(argv):
    dataset = "fashioniq"
    cleaned = [argv[0]]
    skip_next = False

    for index, arg in enumerate(argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue

        if arg == "--dataset":
            if index + 1 >= len(argv):
                raise SystemExit("--dataset requires either fashioniq or cirr")
            dataset = argv[index + 1].lower()
            skip_next = True
            continue

        if arg.startswith("--dataset="):
            dataset = arg.split("=", 1)[1].lower()
            continue

        cleaned.append(arg)

    if dataset not in {"fashioniq", "cirr"}:
        raise SystemExit("--dataset should be either fashioniq or cirr")
    return dataset, cleaned


if __name__ == "__main__":
    os.environ.setdefault("REST_WEIGHT_ROOT", str(Path(__file__).resolve().parents[2] / "weight" / "Composed_Image_Retrieval"))
    dataset, sys.argv = pop_dataset_arg(sys.argv)
    runpy.run_module(dataset, run_name="__main__")
