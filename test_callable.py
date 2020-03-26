import os
import torch

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
        "railsem19": [0.0, 0.0, 0.0],
        "vistas": [80.5423, 91.3162, 81.4312]}

def prepare_img(img0, orig_size, img_mean, img_norm):
    w_add_both = 0
    h_add_both = 0
    if img0.shape[0] - 9 < orig_size[0] and img0.shape[1] < orig_size[1]: #apply padding, keep image in center
        w_add_both = orig_size[1]-img0.shape[1]
        h_add_both = orig_size[0]-img0.shape[0]
        h_add_both0 = h_add_both 
        if h_add_both < 0: #this removes up to 8 lines at the bottom so that 1024/1025 height models can work for 1032 height inputs without scaling
            print("old_dims:",img0.shape)
            img0 = img0[:h_add_both,:,:]
            print("new_dims0:",img0.shape)
            h_add_both0 = 0
        img = np.pad(img0,pad_width=[(h_add_both0//2,h_add_both0-h_add_both0//2),(w_add_both//2,w_add_both-w_add_both//2),(0,0)],mode='constant', constant_values=0)
        if h_add_both < 0:
            print("new_dims1:",img.shape)
    else:
        img = misc.imresize(img0, orig_size)  # uint8 with RGB mode
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img -= img_mean
    if img_norm:
        img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    return img, w_add_both, h_add_both

def decode_segmap(temp, colors):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(len(colors)):
        r[temp == l] = colors[l][0]
        g[temp == l] = colors[l][1]
        b[temp == l] = colors[l][2]
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")].replace('icenet','icnet')
    corr_name = {"psp":"pspnet"}
    model_name = corr_name.get(model_name,model_name)
    model_name_shrt = model_name[:min(5,len(model_name))].lower()
    allfiles = [args.img_path]
    if os.path.isdir(args.img_path):
        allfiles = files_in_subdirs(args.img_path)
        
    if args.inp_dim == None:
        img = misc.imread(allfiles[0])
        orig_size = img.shape[:-1]
    else:
        orig_size = [int(dim) for dim in args.inp_dim.split("x")]
        orig_size = [orig_size[1],orig_size[0]]

    calc_pred_quality = args.check_options
    # Setup image
    if not calc_pred_quality:
        print("Read Input Image from : {}, model {}, inp.sz:".format(args.img_path,model_name_shrt), orig_size)
        from tqdm import tqdm_notebook as tqdm
    else:
        tqdm = lambda *i, **kwargs: i[0]  # pylint:disable=invalid-name
    
    img_mean = mean_rgb[args.version]
    colors = []
    if len(args.vis_dataset) > 0:
        data_loader_vis = get_loader(args.vis_dataset)
        loader_vis = data_loader_vis(root=None, is_transform=True, version=args.version, img_size=orig_size, img_norm=args.img_norm, test_mode=True)
        colors = loader_vis.colors
    

    # Setup Model
    model_dict = {"arch": model_name, "input_size":tuple(orig_size)}
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    potential_n_class = ['classif_conv.weight', 'classification.weight']
    
    #automatically detect number of classes
    n_classes = 19
    for p in potential_n_class:
        if p in state:
            n_classes = state[p].shape[0]
            break
    
    model = get_model(model_dict, n_classes, version=None)
    
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    all_lab = set(range(n_classes))
    outdir = args.out_path
    outp_is_dir = max(outdir.find('.jpg'), outdir.find('.png')) < 0
    
    num_files = len(allfiles)
    all_hists = np.zeros((num_files,n_classes),dtype=np.int32)
    all_qual = np.zeros((num_files),dtype=np.float)
    res_idx = 0
    
    if outp_is_dir:
        outdir += '/'
    if not os.path.exists(os.path.dirname(outdir)):
        os.makedirs(os.path.dirname(outdir))
    for f in tqdm(allfiles, "Calculating predictions..."):
        outname = outdir
        if outp_is_dir:
            outfile0 = os.path.basename(f).replace('.jpg','.png')
            if args.check_options:
                outfile0 = args.version + "_" + str(args.img_norm) + outfile0
            outname = os.path.join(os.path.dirname(outdir), outfile0)
        if not calc_pred_quality and os.path.exists(outname):
            continue
        img =misc.imread(f)
        restore_dim = img.shape[:-1]
        img, w_add_both, h_add_both = prepare_img(img, orig_size, img_mean, args.img_norm)
        with torch.no_grad():
            img = torch.from_numpy(img).float()
            images = img.to(device)
            outputs = model(images)
            pred_qual = None
            if calc_pred_quality:
                pred_qual = np.squeeze(outputs.data.max(1)[0].cpu().numpy(), axis=0)
            pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
            
            if w_add_both > 0:
                pred = pred[:,w_add_both//2:-(w_add_both//2)]
            if h_add_both > 0:
                pred = pred[h_add_both//2:-(h_add_both//2),:]
            if h_add_both < 0:
                add_invalids = np.ones((-h_add_both,pred.shape[1]), dtype = pred.dtype)*255
                pred = np.vstack((pred,add_invalids))
            if model_name_shrt in ["pspne", "icnet", "frrnb"]:
                pred = pred.astype(np.float32)
                if calc_pred_quality:
                    pred_qual = pred_qual.astype(np.float32)
                # float32 with F mode, resize back to restore_dim
                pred = misc.imresize(pred, restore_dim, "nearest", mode="F")
                # no scaling for pred_qual necessary (this is for statistical comparisions)
        
        missings = sorted(list(all_lab-set(np.unique(pred))))
        pred = np.uint8(pred)
        if calc_pred_quality:
            hist0 = np.bincount(pred.flatten())
            n_hist = min(n_classes,hist0.shape[0])
            all_hists[res_idx, 0:n_hist] = hist0[0:n_hist]
            all_qual[res_idx] = float(np.mean(pred_qual))
            res_idx += 1
        if not calc_pred_quality or (len(colors) > 0 and not os.path.exists(outname)):
            if len(colors) > 0:
                pred = decode_segmap(pred, colors)
            pilimg.fromarray(pred).save(outname)
            if len(allfiles) < 4:
                print("Segmentation Pred. Saved at: {}; missing classes:".format(outname), missings)
                
    if calc_pred_quality:
        all_hists = np.mean(all_hists[0:res_idx], axis=0)
        all_qual = all_qual[0:res_idx]
        all_hists = all_hists / np.max(all_hists)
        all_qual = all_qual / np.max(all_qual)

        n0 = np.unique(all_hists)
        diff_of_diffs = 0
        if len(n0) > 2:
            diff_of_diffs = np.sum(n0[1:-1]-n0[0:-2])
        return (np.mean(all_hists), diff_of_diffs), (float(np.mean(all_qual,axis=0)), float(np.std(all_qual,axis=0))), all_hists
        

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
        "--vis_dataset",
        nargs="?",
        type=str,
        default="",
        help="False-colour rgb mapping to use for results (cityscapes or railsem19; empty will return original label ids in uint8)",
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
    parser.add_argument(
        "--check-options",
        dest="check_options",
        action="store_true",
        help="Check all possible image normalization settings to find correct one",
    )
        
    parser.set_defaults(img_norm=True, check_options=False)

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
        
    if args.check_options:
        best_score = -1.0
        best_params = None
        for name in mean_rgb.keys():
            if name == 'railsem19':
                continue
            args.version = name
            for imgn in range(2):
                args.img_norm = (imgn > 0)
                
                class_distr, conf, hist = test(args)
                q_score = class_distr[0]*class_distr[1]
                if len(args.vis_dataset) > 0:
                    print("Checked %s (n. %i):" %( args.version, args.img_norm), q_score, class_distr, conf)
                if q_score > best_score:
                    best_score = q_score
                    best_params = (best_score, args.version, args.img_norm, class_distr, conf)
        print("Best version:", best_params)   
    else:
        test(args)
    return 0

if __name__ == "__main__":
    sys.exit(main_test(sys.argv[1:]))
