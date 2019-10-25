import os
import torch
from tqdm import tqdm_notebook as tqdm
import argparse, glob
import numpy as np
import scipy.misc as misc
from PIL import Image as pilimg
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

def files_in_subdirs(start_dir, pattern = ["*.png","*.jpg","*.jpeg"]):
    files = []
    for p in pattern:
        for dir,_,_ in os.walk(start_dir):
            files.extend(glob.glob(os.path.join(dir,p)))
    return files

mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
        "railsem19": [0.0, 0.0, 0.0]}

def prepare_img(img0, orig_size, img_mean, img_norm):
    img = misc.imresize(img0, orig_size)  # uint8 with RGB mode
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img -= img_mean
    if img_norm:
        img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    return img

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    allfiles = [args.img_path]
    if os.path.isdir(args.img_path):
        allfiles = files_in_subdirs(args.img_path)
        
    # Setup image
    print("Read Input Image from : {}".format(args.img_path))

    if args.inp_dim == None:
       img = misc.imread(allfiles[0])
       orig_size = img.shape[:-1]
    else:
       orig_size = [int(dim) for dim in args.inp_dim.split("x")]

    print(orig_size)
    data_loader = get_loader(args.dataset)
    img_mean = mean_rgb[args.version]
    loader = data_loader(root=None, is_transform=True, version=args.version, img_size=orig_size, img_norm=args.img_norm, test_mode=True)
    
    n_classes = loader.n_classes
  
    # Setup Model
    model_dict = {"arch": model_name, "input_size":tuple(orig_size)}
    model = get_model(model_dict, n_classes, version=args.dataset)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
   
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    all_lab = set(range(19))
    outdir = args.out_path
    outp_is_dir = max(outdir.find('.jpg'), outdir.find('.png')) < 0
    if outp_is_dir:
        outdir += '/'
    if not os.path.exists(os.path.dirname(outdir)):
        os.makedirs(os.path.dirname(outdir))
    for f in tqdm(allfiles, "Calculating predictions..."):
        outname = outdir
        if outp_is_dir:
            outname = os.path.join(os.path.dirname(outdir), os.path.basename(f).replace('.jpg','.png'))
        if os.path.exists(outname):
            continue
        img =misc.imread(f)
        img = prepare_img(img, orig_size, img_mean, args.img_norm)# prepare_img(img, orig_size, model_name, loader, args.img_norm)
        with torch.no_grad():
            img = torch.from_numpy(img).float()
            images = img.to(device)
            outputs = model(images)
            pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
            if model_name[:min(5,len(model_name))] in ["pspne", "icnet"]:
                pred = pred.astype(np.float32)
                # float32 with F mode, resize back to orig_size
                pred = misc.imresize(pred, orig_size, "nearest", mode="F")
        
        #print(pred.shape)
        #decoded = loader.decode_segmap(pred)
        missings = sorted(list(all_lab-set(np.unique(pred))))
        pilimg.fromarray(np.uint8(pred)).save(outname)
        if len(allfiles) < 4:
            print("Segmentation Pred. Saved at: {}; missing classes:".format(outname), missings)

def main_test(arg0):
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="pascal",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )
    parser.add_argument(
        "--inp_dim",
        nargs="?",
        type=str,
        default=None,
        help="Fix input/output dimensions (e.g. 1920x1080); default: use dimensions of first test image",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="cityscapes",
        help="Image normalization to use ['pascal, cityscapes']",
    )
    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--dcrf",
        dest="dcrf",
        action="store_true",
        help="Enable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--no-dcrf",
        dest="dcrf",
        action="store_false",
        help="Disable DenseCRF based post-processing | \
                              False by default",
    )
    parser.set_defaults(dcrf=False)

    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--out_path", nargs="?", type=str, default=None, help="Path of the output segmap"
    )
    args = parser.parse_args(arg0)
    test(args)
    return 0

if __name__ == "__main__":
    sys.exit(main_test(sys.argv[1:]))
