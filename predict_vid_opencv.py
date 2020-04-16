import os
import torch

import argparse, glob
import numpy as np
import cv2
from PIL import Image as pilimg

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict
from tqdm import tqdm_notebook as tqdm
from export_onnx import torch_uint8_to_float, torch_uint8_to_float_normed, torch_downsample_to_size, torch_gaussian_blur, torch_return_uint8_argmax

import os, sys, re, fnmatch
def walk_maxd(root, maxdepth):
    dirs, nondirs = [], []
    for name in os.listdir(root):
        (dirs if os.path.isdir(os.path.join(root, name)) else nondirs).append(name)
    yield root, dirs, nondirs
    if maxdepth > 1:
        for name in dirs:
            for x in walk(os.path.join(root, name), maxdepth-1):
                yield x
                
def glob_dirs_ic(root, pattern= ['*.jpg','*.png','*.jpeg'], maxdepth = 1):
    reg_expr = re.compile('|'.join(fnmatch.translate(p) for p in pattern), re.IGNORECASE)
    result = []
    for root, dirs, files in walk_maxd(root=root, maxdepth=maxdepth):
        result += [os.path.join(root, j) for j in files if re.match(reg_expr, j)]
    return result

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
            img0 = img0[:h_add_both,:,:]
            h_add_both0 = 0
        img = np.pad(img0,pad_width=[(h_add_both0//2,h_add_both0-h_add_both0//2),(w_add_both//2,w_add_both-w_add_both//2),(0,0)],mode='constant', constant_values=0)
    else:
        #this framework resizes using PIL to keep comparability with previous results; PIL resize is distinctively different from opencv; especially for downscaling (applies custom gauss filtering for anti-aliasing!)
        img = np.array(pilimg.fromarray(img0).resize((orig_size[1],orig_size[0]), pilimg.BILINEAR)) # uint8 BGR
        #img = cv2.resize(img0, (orig_size[1],orig_size[0]),0,0,cv2.INTER_LINEAR)  # uint8 BGR
    return img, w_add_both, h_add_both

def decode_segmap(temp, colors):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(len(colors)):
        r[temp == l] = colors[l][2]
        g[temp == l] = colors[l][1]
        b[temp == l] = colors[l][0]
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb

def get_video_frame(vcap, skip_idx=0):
    for i in range(skip_idx):
        vcap.grab() #allows skipping of frames
    video_idx = int(vcap.get(1))
    ret, img = vcap.read()
    if not ret or img is None:
        return None, video_idx
    return img, video_idx

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")].replace('icenet','icnet')
    corr_name = {"psp":"pspnet"}
    model_name = corr_name.get(model_name,model_name)
    model_name_shrt = model_name[:min(5,len(model_name))].lower()

    src_is_vid = args.inp_path.find('.mp4') > 0 #support video files or folders with images
    
    if src_is_vid:
        vcap = cv2.VideoCapture(args.inp_path)  
        fps = vcap.get(cv2.CAP_PROP_FPS)
        if fps < 0.1:
            fps = 25
        max_frms = (int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)) // args.decimate) +2
        im0, video_idx = get_video_frame(vcap)   
        if im0 is None:
            print("Cannot open video " + args.inp_path)
            return -3
    else:
        if '.png' in args.inp_path or '.jpg' in args.inp_path:
            all_frames = [args.inp_path]
        else:
            all_frames = files_in_subdirs(args.inp_path, pattern = ["*.png","*.jpg","*.jpeg"])
        if len(all_frames) == 0:
            print("Found no images in directory " + args.inp_path)
            return -4
        im0 = cv2.imread(all_frames[0])
        if im0 is None:
            print("Cannot open image " + all_frames[0])
            return -5
        if args.decimate > 1:
            all_frames = all_frames[::args.decimate]
        max_frms = len(all_frames)
        
    restore_dim = (im0.shape[1],im0.shape[0])

    if args.inp_dim == None:
        orig_size = restore_dim
    else:
        orig_size = [int(dim) for dim in args.inp_dim.split("x")]
        orig_size = [orig_size[1],orig_size[0]]

    # Setup image
    print("Reading {} frames from input {}, model {}, model inp.sz, inp sz:".format(str(max_frms), args.inp_path,model_name_shrt), orig_size, restore_dim)
   
    img_mean = mean_rgb[args.version]
    colors = []
    blend_frm = args.vis_dataset.split(';')[-1].lower().find('blend') >= 0
    if len(args.vis_dataset) > 0:
        data_loader_vis = get_loader(args.vis_dataset.split(';')[0])
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
    if args.img_norm:
        model_fromuint8 = torch.nn.Sequential(torch_uint8_to_float_normed(), model, torch_return_uint8_argmax())
    else:
        model_fromuint8 = torch.nn.Sequential(torch_uint8_to_float(), model, torch_return_uint8_argmax())
    model_fromuint8.eval()
    model_fromuint8.to(device)
    
    all_lab = set(range(n_classes))
    outp_path = args.out_path
    outp_is_dir = max(outp_path.find('.mp4'), outp_path.find('.divx')) < 0
    
    if src_is_vid and outp_is_dir:
        outp_path += '/vis_'+args.inp_path+'.mp4'
    if not os.path.exists(os.path.dirname(outp_path)):
        os.makedirs(os.path.dirname(outp_path))
    if src_is_vid:
        cap_out = cv2.VideoWriter(outp_path,cv2.VideoWriter_fourcc(*'MP4V'), fps // args.decimate, (restore_dim[1],restore_dim[0]))
    access_idx = 0
    for f in tqdm(range(max_frms), "Calculating predictions..."):
        if im0 is None:
            break
        if not src_is_vid:
            restore_dim = (im0.shape[1],im0.shape[0])
        if orig_size[0] != restore_dim[1] or orig_size[1] != restore_dim[0]:
            img, w_add_both, h_add_both = prepare_img(im0, orig_size, img_mean, args.img_norm)
        else:
            img, w_add_both, h_add_both = im0, 0, 0
        with torch.no_grad():
            img = torch.from_numpy(img)
            images = img.to(device)
            outputs = model_fromuint8(images)
            pred = outputs.cpu().numpy()
            
            if w_add_both > 0:
                pred = pred[:,w_add_both//2:-(w_add_both//2)]
            if h_add_both > 0:
                pred = pred[h_add_both//2:-(h_add_both//2),:]
            if h_add_both < 0:
                add_invalids = np.ones((-h_add_both,pred.shape[1]), dtype = pred.dtype)*255
                pred = np.vstack((pred,add_invalids))
            #resize back to restore_dim
            pred = cv2.resize(pred, restore_dim, interpolation=cv2.INTER_NEAREST)
            
        outext = '.png'
        if len(colors) > 0:
            outext = '.jpg'
            pred = decode_segmap(pred, colors)
            if blend_frm:
                pred = cv2.addWeighted(im0, 0.5, pred, 0.5, 0.0)
        
        if src_is_vid:
            cap_out.write(pred)
            im0, access_idx = get_video_frame(vcap, skip_idx=(args.decimate-1))
        else:
            outfile0 = os.path.basename(all_frames[access_idx]).replace('.jpg','').replace('.png','')
            outname = os.path.join(os.path.dirname(outp_path), outfile0+outext)
            cv2.imwrite(outname, pred)
            access_idx += 1
            if access_idx >= len(all_frames):
                im0 = None
            else:
                im0 = cv2.imread(all_frames[access_idx])
    if src_is_vid:
        cap_out.release()
        vcap.release()
    return images, pred
            

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
        
    parser.set_defaults(img_norm=True)


    parser.add_argument(
        "--inp_path", nargs="?", type=str, default=None, help="Path of the input video or directory"
    )
    parser.add_argument(
        "--out_path", nargs="?", type=str, default=None, help="Path of the output video or directory"
    )
    parser.add_argument(
        "--decimate", nargs="?", type=int, default=1, help="Decimation fps (take only every x frame; also applies to directies)"
    )
    
    args = parser.parse_args(arg0)
    return test(args)

if __name__ == "__main__":
    sys.exit(main_test(sys.argv[1:]))
