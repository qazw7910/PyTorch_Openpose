import argparse
import contextlib
import time
from pathlib import Path
from typing import Sequence, Dict

import torch
from torch2trt import torch2trt, TRTModule

from keypoint_models.models import KeypointLSTM


def main(args: Sequence[str] = None):
    # setup paths and check existence
    args = get_arg_parser().parse_args(args)
    pth_path: Path = args.original_model
    if not pth_path.is_file():
        raise FileNotFoundError(pth_path)

    out: Path = args.trt_out
    if not out.is_dir():
        if out.exists():
            raise NotADirectoryError(out)
        else:
            raise FileNotFoundError(out)

    mode = args.original_model_mode

    with eval_time("load_model"):
        weights = torch.load(pth_path)
        if mode == "pytorch":
            model = KeypointLSTM(input_size=30, hidden_size=64, num_layers=1, num_classes=2, batch_first=True)
        elif mode == "trt":
            model = TRTModule()

        model.load_state_dict(weights)
        model.eval().cuda()

    # create example data
    x = torch.ones((1, 20, 30)).cuda()

    with eval_time("convert_model"):
        if mode == "pytorch":
            model_trt = torch2trt(
                model, [x],
                use_onnx=True,
                fp16_mode=True,
                int8_mode=True,
                max_workspace_size=1 << 30
            )
        else:
            model_trt = TRTModule()
            model_trt.load_state_dict(weights)
        model_trt.eval()

    time_record = {}

    with torch.no_grad():
        print("normal model: ", model)
        with eval_time("normal", time_record=time_record):
            y = model(x)
            for _ in range(999):
                y += model(x)

        print("trt model: ", model_trt)
        with eval_time("trt", time_record=time_record):
            y_trt = model_trt(x)
            for _ in range(999):
                y_trt += model_trt(x)

    print(torch.sqrt(torch.sum((y - y_trt) ** 2)))
    print(time_record['normal'] / time_record['trt'])
    print(y)
    print(y_trt)

    if mode == "pytorch":
        torch.save(model_trt.state_dict(), out / pth_path.name)


@contextlib.contextmanager
def eval_time(name: str, log_func=print, time_record: Dict[str, float] = None):
    start = time.time()
    yield
    duration = time.time() - start
    log_func(f"execution time of {name}: {duration: .6f}")
    if time_record is not None:
        time_record[name] = duration


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert pytorch model to trt model and evaluate execution time of original and converted model"
    )
    parser.add_argument("original_model", type=Path)
    parser.add_argument("-o", "--trt_out", type=Path, default="trt_model")
    parser.add_argument("-m", "--original_model_mode", type=str, default="pytorch", choices=['pytorch', 'trt'])

    return parser


if __name__ == '__main__':
    main()
