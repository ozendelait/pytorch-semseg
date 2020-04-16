import os

import argparse, glob
import numpy as np
import cv2
from PIL import Image as pilimg
from tqdm import tqdm_notebook as tqdm
import onnxruntime
import os, sys, re, fnmatch

colors_rs19 = [ [128, 64, 128], [244, 35, 232],    [70, 70, 70],  [192, 0, 128], [190, 153, 153],
               [153, 153, 153], [250, 170, 30],   [220, 220, 0], [107, 142, 35], [152, 251, 152],
                [70, 130, 180],  [220, 20, 60], [230, 150, 140],    [0, 0, 142], [0, 0, 70],
                  [90, 40, 40],   [0, 80, 100],   [0, 254, 254],    [0, 68, 63]]
colors_cs = [ [128, 64, 128], [244, 35, 232],  [70, 70, 70], [102, 102, 156], [190, 153, 153],
             [153, 153, 153], [250, 170, 30], [220, 220, 0],  [107, 142, 35], [152, 251, 152],
               [0, 130, 180],  [220, 20, 60],   [255, 0, 0],     [0, 0, 142], [0, 0, 70],
                [0, 60, 100],   [0, 80, 100],   [0, 0, 230],   [119, 11, 32]]


def files_in_subdirs(start_dir, pattern = ["*.png","*.jpg","*.jpeg"]):
    files = []
    for p in pattern:
        for dir,_,_ in os.walk(start_dir):
            files.extend(glob.glob(os.path.join(dir,p)))
    return files

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

def prepare_img(img0, orig_size):
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
        #img = cv2.resize(img0, (orig_size[1],orig_size[0]))  # uint8 with RGB mode
        img = np.array(pilimg.fromarray(img0).resize((orig_size[1],orig_size[0]), pilimg.BILINEAR))
    return img, w_add_both, h_add_both

def test(args):
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
    
#sess.set_providers(['CPUExecutionProvider'])
    ort_session = onnxruntime.InferenceSession(args.model_path)
    orig_size = (ort_session.get_inputs()[0].shape[0],ort_session.get_inputs()[0].shape[1])

    # Setup image
    print("Reading {} frames from input {}, model {}, model inp.sz, inp sz, providers:".format(str(max_frms), args.inp_path,args.model_path), orig_size, restore_dim, ort_session.get_providers())
   
    colors = []
    blend_frm = args.vis_dataset.split(';')[-1].lower().find('blend') >= 0
    if len(args.vis_dataset) > 0:
        colors = colors_rs19 if "rail" in args.vis_dataset.lower() else colors_cs
    
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
            img, w_add_both, h_add_both = prepare_img(im0, orig_size)
        else:
            img, w_add_both, h_add_both = im0, 0, 0
        ort_inputs = {ort_session.get_inputs()[0].name: img}
        ort_outs = ort_session.run(None, ort_inputs)
        pred = ort_outs[0]

        if w_add_both > 0:
            pred = pred[:,w_add_both//2:-(w_add_both//2)]
        if h_add_both > 0:
            pred = pred[h_add_both//2:-(h_add_both//2),:]
        if h_add_both < 0:
            add_invalids = np.ones((-h_add_both,pred.shape[1]), dtype = pred.dtype)*255
            pred = np.vstack((pred,add_invalids))
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
    return im0
            

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
    test(args)
    return 0

if __name__ == "__main__":
    sys.exit(main_test(sys.argv[1:]))
