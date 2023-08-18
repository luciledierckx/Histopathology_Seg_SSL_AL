"evaluate: save test metrics for all checkpoints in a csv file"

import argparse
import os
import pathlib
import re
import sys

import pandas as pd
import torch

import dataset
import query_strategies
from models import UNet
from query_strategies.strategy import Strategy

IMAGE_SIZE = 256
DATASET = "glas"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser(prog="evaluate")
    parser.add_argument(
        "destination", type=str, default="results.csv", help="CSV file for the results", nargs="?"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./save", help="dir where all saved model are"
    )
    parser.add_argument(
        "--allow-duplicates", type=bool, default=False, help="allow duplicates while merging"
    )
    args = parser.parse_args(sys.argv[1:])

    dest_csv = pathlib.Path(args.destination)

    save_dir = pathlib.Path(args.save_dir)
    if not save_dir.exists() or not save_dir.is_dir():
        raise ValueError(f"save-dir={save_dir!r} should be an existing directory")

    allow_duplicates: bool = args.allow_duplicates

    return dest_csv, save_dir, allow_duplicates


class _Wrapper(torch.nn.Module):
    "wrapper module, see load_module"

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def load_module(path: pathlib.Path, module: torch.nn.Module):
    "load module while fixing inconsistent state-dict keys"

    state_dict = torch.load(path, map_location=DEVICE)

    # check if module is wrapped
    example_key: str = next(iter(state_dict.keys()))
    wrapped = example_key.startswith("module.")

    if wrapped:
        wrapped_module = _Wrapper(module)
        wrapped_module.load_state_dict(state_dict)
        return wrapped_module

    module.load_state_dict(state_dict)
    return module


def to_float(val: "float | torch.Tensor"):
    "make sure the value is a float"
    if isinstance(val, torch.Tensor):
        return float(val.item())
    if isinstance(val, float):
        return val
    raise TypeError(f"{val=!r} is not recognized")


def load_and_eval(
    file: pathlib.Path,
    strategy: str,
):
    handler = dataset.get_handler(DATASET)
    strategy_args = argparse.Namespace(
        transform_te=None,
        loader_te_args={
            "batch_size": 8,
            "num_workers": 0,
        },
        save_image_freq=-1,
    )

    # load dataset
    _, Y_tr, _, _, X_te, Y_te, _ = dataset.get_dataset(
        name=DATASET,
        path="./datasets",
        doFullySupervized=True,  # doesn't matter, we won't look at Y_tr
    )

    # load the model
    net = load_module(file, UNet(n_class=2))
    net = net.to(DEVICE)

    # load the strategy
    strategy: Strategy = query_strategies.__dict__[strategy](
        X=None,
        Y=Y_tr,  # required for self.n_pool
        X_val=None,
        Y_val=None,
        idxs_lb=None,
        net=net,
        handler=handler,
        args=strategy_args,  # required for predict
    )

    dsc, mcc, loss = strategy.predict(X_te, Y_te)

    return to_float(dsc), to_float(mcc), to_float(loss)


def eval_all_checkpoints(save_dir: pathlib.Path):
    pattern_group = (
        r"tensorboard_([^_]+)_([^_]+)_proRemoveGland(0.[0-9]+)_"
        r"doFullySup(True|False)_nepoch([0-9]+)_([0-9]+)_best_model.pkl"
    )
    pattern_search = (
        "tensorboard_*_proRemoveGland*_doFullySup*_nepoch*_*_best_model.pkl"
    )

    matcher = re.compile(pattern_group)
    columns = (
        "strategy",
        "query_strategy",
        "p_removal",
        "fully_supervised",
        "n_epochs",
        "round",
        "dice",
        "mcc",
        "loss",
    )
    rows: list[tuple[str, str, float, bool, int, int, float]] = []

    for file in save_dir.glob(pattern_search):
        # find the model
        if not (match := matcher.match(file.name)):
            raise ValueError(f"invalid {file=!r}")

        # model = match.group(0)
        rows.append(
            (
                match.group(1),
                match.group(2),
                float(match.group(3)),
                match.group(4).lower() == "true",
                int(match.group(5)),
                int(match.group(6)),
                *load_and_eval(file, match.group(1)),
            )
        )

    return pd.DataFrame(rows, columns=columns)


def concat(main: pd.DataFrame, new: pd.DataFrame, allow_duplicates: bool) -> pd.DataFrame:
    "concat two dataframe into one"

    if list(main.columns) != list(new.columns):
        raise ValueError(f"inconsistent columns, unable to concat ({main=!r} {new=!r})")

    keys = ["strategy", "query_strategy", "p_removal", "n_epochs", "round"]

    # check for duplicates in original
    if not allow_duplicates and main[keys].duplicated().any():
        raise ValueError("original dataset contains")

    concat = pd.concat(
        [main, new],
        ignore_index=True,
    )

    # check for duplicate rows in the concatenated
    if not allow_duplicates and concat[keys].duplicated().any():
        raise ValueError("concat would create duplicate values")

    return concat


###############################################################################
###############################################################################


def main():
    dest_csv, checkpoint_dir, allow_duplicates = get_args()

    # make sure the test patches exist
    if not os.path.exists("./datasets/glas/test_patches/"):
        print("creating test patches")
        command = (
            "python prepare_data/removeAnn_extractPatches.py --labels_dir "
            "datasets/glas/test_labels/ --imgs_dir datasets/glas/test_samples/ "
            "--output_dir datasets/glas/test_patches/ --probRemove 0 "
            f"--patchSize {IMAGE_SIZE}"
        )
        os.system(command)
    else:
        print("found test patches")

    dataframe = eval_all_checkpoints(checkpoint_dir)

    # auto merge if the file already exists
    if dest_csv.exists():
        main_df = pd.read_csv(dest_csv)
        dataframe = concat(main_df, dataframe, allow_duplicates)

    dataframe.to_csv(dest_csv, index=False)


if __name__ == "__main__":
    main()
