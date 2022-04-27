import contextlib
import shutil
import time
from io import BytesIO
from pathlib import Path

import onnx
import torch
from torch2trt import torch2trt, TRTModule

from keypoint_models.models import KeypointLSTM


def main():
    # pth_path = Path("trt_model/fall_20_fps.pth")
    # mode = "trt"

    pth_path = Path("pt_model/fall_20_fps.pth")
    mode = "pytorch"

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

    with torch.no_grad():
        print("normal model: ", model)
        with eval_time("normal"):
            y = model(x)
            for _ in range(999):
                y += model(x)

        print("trt model: ", model_trt)
        with eval_time("trt"):
            y_trt = model_trt(x)
            for _ in range(999):
                y_trt += model_trt(x)


    print(torch.max(torch.abs(y - y_trt)))
    print(y)
    print(y_trt)

    if mode == "pytorch":
        torch.save(model_trt.state_dict(), "trt_model/fall_20_fps.pth")


@contextlib.contextmanager
def eval_time(name: str, log_func=print):
    start = time.time()
    yield
    log_func(f"execution time of {name}: {time.time() - start: .6f}")


if __name__ == '__main__':
    main()
