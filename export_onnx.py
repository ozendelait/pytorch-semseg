import os
import torch
import torch.nn.functional as F
from collections import OrderedDict
import argparse

from ptsemseg.models import get_model
from ptsemseg.utils import convert_state_dict

import numpy as np    
#based on https://github.com/ducha-aiki/ucn-pytorch/blob/master/Utils.py ; commit de5bdec from Oct 4, 2017
class torch_gaussian_blur(torch.nn.Module):

    def __init__(self, kern_sz = 3):
        super(torch_gaussian_blur, self).__init__()
        self.pad = int(np.floor(kern_sz/2.0))
        self.weight = self.calculate_weights(kern_sz = kern_sz)
        
    def circular_gauss_kernel(self, kern_sz, circ_zeros = False, norm = True):
        kern_sz_half = kern_sz / 2.0;
        r2 = float(kern_sz_half*kern_sz_half)
        sigma2 = 0.9 * r2;
        sigma = np.sqrt(sigma2)
        x = np.linspace(-kern_sz_half,kern_sz_half,kern_sz)
        xv, yv = np.meshgrid(x, x, sparse=False, indexing='xy')
        distsq = (xv)**2 + (yv)**2
        kernel = np.exp(-( distsq/ (sigma2)))
        if circ_zeros:
            kernel *= (distsq <= r2).astype(np.float32)
        if norm:
            kernel /= np.sum(kernel)
        #if is_rgb:
        #    kernel = np.tile(kernel[None, :, :], [3, 1, 1])
        return kernel
    
    def calculate_weights(self, kern_sz):
        kernel = self.circular_gauss_kernel(kern_sz = kern_sz, circ_zeros = False, norm = True)
        return torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0).contiguous();
    
    def forward(self, x):
        b,c,h,w = x.shape
        x = F.pad(x, (self.pad,self.pad,self.pad,self.pad), 'replicate')
        hp, wp = x.shape[2], x.shape[3]
        return F.conv2d(x.reshape(b*c,1,hp,wp),self.weight.to(x.device), padding = 0).reshape(b,c,h,w)

#optimized uint8 HWC BGR -> float 1CHW  directly in CUDA/pytorch code
class torch_downsample_to_size(torch.nn.Module):
    def __init__(self, trg_size, trg_mode):
        super(torch_downsample_to_size, self).__init__()
        self.trg_size = trg_size
        self.trg_mode = trg_mode
        
    def forward(self, x):
        #return imresize0(x, resized_shape=self.trg_size, output_crop_shape=None, darknet=False, edge=True, axis=2)
        return torch.nn.functional.interpolate(x, self.trg_size, mode= 'bilinear', align_corners=False)

class torch_uint8_to_float(torch.nn.Module):
    def __init__(self):
        super(torch_uint8_to_float, self).__init__()
    def forward(self, x):
        #based on https://gist.github.com/xvdp/149e8c7f532ffb58f29344e5d2a1bee0
               #HWC -> CHW          uint8 -> float     CHW-> 1CHW
        return x.permute(2,0,1).to(dtype=torch.float).unsqueeze(0).contiguous()

class torch_uint8_to_float_normed(torch.nn.Module):
    def __init__(self):
        super(torch_uint8_to_float_normed, self).__init__() 
    def forward(self, x):
        #based on https://gist.github.com/xvdp/149e8c7f532ffb58f29344e5d2a1bee0
                 #HWC -> CHW          uint8 -> float   normalize  CHW-> 1CHW
        return  (x.permute(2,0,1).to(dtype=torch.float) / 255.).unsqueeze(0).contiguous()
        
def get_num_classes(state):
    # Setup Model
    potential_n_class = ['classif_conv.weight', 'classification.weight']
    #automatically detect number of classes
    n_classes = 19
    for p in potential_n_class:
        if p in state:
            n_classes = state[p].shape[0]
            break
    return n_classes
    
def main_export_onnx(arg0):
    parser = argparse.ArgumentParser(description="Program to convert model to onnx; use converter first if you need normalization or mean shifts.\nParams:")
    parser.add_argument(
        "--model_path", nargs="?", type=str, default="frrnB_cityscapes_best_model_miou63.pkl", help="Path to the saved model")
    parser.add_argument(
        "--out_path", nargs="?", type=str, default="", help="Path for saving transformed model, default: inp.onnx")
    parser.add_argument(
        "--inp_dim", nargs="?", type=str, default="1920x1080", help="Fixed input/output dimensions as WxH; default: 1920x1080")
    parser.add_argument(
        "--img_norm", dest="img_norm", action="store_true", help="Source model expectesn scaling from [0;1] (target ONNX [0;255])",
    )
    parser.set_defaults(img_norm=False)
    args = parser.parse_args(arg0)

    if len(args.out_path) == 0:
        args.out_path = args.model_path.replace('.pkl','.onnx')
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_size = [int(dim) for dim in args.inp_dim.split("x")]
    orig_size = [orig_size[1],orig_size[0]]
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")].replace('icenet','icnet')
    model_name = {"psp":"pspnet"}.get(model_name,model_name)
    model_dict = {"arch": model_name, "input_size":tuple(orig_size)}
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model = get_model(model_dict, get_num_classes(state), version=None)
    model.load_state_dict(state)
    model.eval()
    if args.img_norm:
        model_fromuint8 = torch.nn.Sequential(torch_uint8_to_float_normed(),model).to(device)
    else:
        model_fromuint8 = torch.nn.Sequential(torch_uint8_to_float(),model).to(device)
    dummy_input = torch.zeros((orig_size[0], orig_size[1], 3), dtype = torch.uint8).to(device)
    
    with torch.no_grad():
        torch.onnx.export(model_fromuint8, dummy_input, args.out_path, input_names=['input_bgr_img'], opset_version=11, verbose=False)
    
    return 0

if __name__ == "__main__":
    sys.exit(main_export_onnx(sys.argv[1:]))
