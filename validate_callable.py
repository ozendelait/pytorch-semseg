import yaml, sys, os
import torch
import argparse
import timeit
import numpy as np

from torch.utils import data
from tqdm import tqdm_notebook as tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict,recursive_glob
import scipy.misc as m

torch.backends.cudnn.benchmark = True

def filepath_to_dict_id(f):
    return os.path.basename(f).replace('.png','').replace("_gtFine_labelTrainIds","").replace("_leftImg8bit","")

def dict_gtfiles_ids(start_dir, pattern = "png"):
    files = recursive_glob(start_dir, pattern)
    return {filepath_to_dict_id(f):f for f in files}

def validate(cfg, args):
    offline_res={}
    #print("device0:",len(args.offline_res), cfg["training"]["batch_size"], len(args.offline_res))
    if len(args.offline_res) > 0:
        offline_res = dict_gtfiles_ids(args.offline_res)
        if len(offline_res) == 0:
            print("Error: No potential result ids found in folder "+args.offline_res)
            return [], [], []
        device = torch.device("cpu")
        cfg["data"]["version"] = "offline_res"
        cfg["training"]["batch_size"] = 1
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]
    val_delta = cfg["data"].get("val_asp_ratio_delta", -1.0)
    if val_delta < 1.0:
        val_delta = -1.0
        
    loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
        is_transform=True,
        img_size=(cfg["model"].get("input_size",[cfg["data"].get("img_rows","same"), "same"])[0] , cfg["model"].get("input_size",["same",cfg["data"].get("img_cols", "same")])[1]),
        version=cfg["data"].get("version","cityscapes"),
        asp_ratio_delta_min = 1.0/val_delta,
        asp_ratio_delta_max = val_delta,
        img_norm=cfg["data"].get("img_norm",True),
    )

    n_classes = loader.n_classes

    valloader = data.DataLoader(loader, batch_size=cfg["training"]["batch_size"], num_workers=8)
    running_metrics = runningScore(n_classes)

    # Setup Model

    model = get_model(cfg["model"], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    total_num_max = len(valloader)
    for i, (images, labels) in enumerate(tqdm(valloader,desc='Validating...')):
        start_time = timeit.default_timer()
        idcheck = "%08i"%i
        if len(offline_res) == 0:
            images = images.to(device)
        if len(offline_res) > 0:
            id_offline_res = filepath_to_dict_id(images[0])
            if not id_offline_res in offline_res:
                print("Warning: id "+ id_offline_res + "not found in offline results!")
                continue
            idcheck = id_offline_res
            pred = m.imread(offline_res[id_offline_res])
            pred = np.array(pred, dtype=np.uint8).reshape(1, pred.shape[0], pred.shape[1])
        elif args.eval_flip:
            outputs = model(images)

            # Flip images in numpy (not support in tensor)
            outputs = outputs.data.cpu().numpy()
            flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
            flipped_images = torch.from_numpy(flipped_images).float().to(device)
            outputs_flipped = model(flipped_images)
            outputs_flipped = outputs_flipped.data.cpu().numpy()
            outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0

            pred = np.argmax(outputs, axis=1)
        else:
            outputs = model(images)
            pred = outputs.data.max(1)[1].cpu().numpy()

        gt = labels.numpy()

        if args.measure_time:
            elapsed_time = timeit.default_timer() - start_time
            print(
                "Inference time \
                  (iter {0:5d}): {1:3.5f} fps".format(
                    i + 1, pred.shape[0] / elapsed_time
                )
            )
        running_metrics.update(gt, pred, [idcheck])

    score, class_iou = running_metrics.get_scores()
    conf_mats = running_metrics.get_confmats()
    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])
    return score, class_iou, conf_mats


def main_val(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--offline_res",
        nargs="?",
        type=str,
        default="",
        help="Path to folder with offline results",
    )
    parser.add_argument(
        "--eval_flip",
        dest="eval_flip",
        action="store_true",
        help="Enable evaluation with flipped image |\
                              True by default",
    )
    parser.add_argument(
        "--no-eval_flip",
        dest="eval_flip",
        action="store_false",
        help="Disable evaluation with flipped image |\
                              True by default",
    )
    parser.set_defaults(eval_flip=True)

    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.set_defaults(measure_time=False, eval_flip= False)

    args = parser.parse_args(argv)

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)
    #need to reduce batchsize to prevent cuda errors
    cfg['training']['batch_size'] = 1

    return validate(cfg, args)

if __name__ == "__main__":
    main_val()
