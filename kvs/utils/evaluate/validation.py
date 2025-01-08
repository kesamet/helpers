"""
Evaluate.
"""
import os
import yaml
from pathlib import Path

from config import CFG
from test import test
from train import set_params
from utils.datasets import create_dataloader
from utils.evaluate.image_extraction import uuid_names


def evaluate(data_dir, camera_ids=[0, 1, 2, 3, 4]):
    """Validation. Returns (P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls))."""
    from utils.serve.detecting import MODEL

    save_dir = "weights/temp"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    params = {
        "weights": os.getenv("modelpath"),
        "cfg": "./models/custom_yolov5s.yaml",
        "data": f"{save_dir}/test.yaml",
        "epochs": 1,
        "batch_size": 16,
        "img_size": [CFG.imgsz, CFG.imgsz],
        "cache_images": True,
        "workers": 0,
        "name": "yolov5s_results",
    }
    opt = set_params(params)

    # Save yaml
    data = {
        "val": [f"{data_dir}/{uuid_names[camera_id]}/images" for camera_id in camera_ids],
        "nc": 2,
        "names": ["Standing_up", "Lying_down"],
    }
    print(data)
    with open(params["data"], "w") as f:
        yaml.dump(data, f)
    
    # Create test dataloader
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader) 
    test_path = data["val"]

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    testloader = create_dataloader(
        test_path, CFG.imgsz, opt.batch_size, int(max(MODEL.stride)), opt,
        hyp=hyp, cache=True, rect=True,
        rank=-1, world_size=1, workers=0)[0]

    # Evaluate
    results, _, _ = test(
        opt.data,
        batch_size=opt.batch_size,
        imgsz=CFG.imgsz,
        model=MODEL,
        single_cls=opt.single_cls,
        dataloader=testloader,
        save_dir=Path(save_dir),
        plots=True,
        log_imgs=0,
    )
    return results


def main():
    modelpath = "weights/neha-yolov5-v14/best.pt"
    data_dir = "weights/dataset"
    camera_ids = [0, 1, 2, 3, 4]

    os.environ["modelpath"] = modelpath
    results = evaluate(data_dir, camera_ids)

    for x, s in zip(["P", "R", "mAP@.5", "mAP@.5-.95"], results[:4]):
        print(f"{x} = {s:.6f}")


if __name__ == "__main__":
    main()
