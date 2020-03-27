import os
import torch

import argparse
import numpy as np

def transform_layer(m0_state, key0, normalization_factor = 1.0, norm_mean = [0.0,0.0,0.0], apply_bgr_flip = True):
    orig_device =  m0_state[key0].get_device()
    w0 = m0_state[key0].cpu().numpy()
    if normalization_factor != 1.0:
        w0 = w0 * normalization_factor
    if apply_bgr_flip:
        if len(w0.shape) == 4 and w0.shape[1] == 3 and w0.shape[0] != 3:
            w0 = np.copy(w0[:,::-1,:,:])
        else:
            print("Warning: unknown position of rgb channel dimension!")
    norm_fact = None
    for c in range(3):
        if norm_mean[c] == 0.0:
            continue
        if len(w0.shape) == 4 and w0.shape[1] == 3 and w0.shape[0] != 3:
            #TODO: find batch norm nodes (where bias is pushed into batch norm)
            w_tmean = np.sum(w0[:,c,:,:] * - norm_mean[c], axis = (1,2)) #assume convolution operation
            if norm_fact is None:
                norm_fact = w_tmean
            else:
                norm_fact += w_tmean
        else:
            print("Warning: unknown position of rgb channel dimension!")
            
    if not norm_fact is None:
        key0_b = key0.replace('.weight','.bias')
        if key0 == key0_b or key0_b not in m0_state:
            print("Warning: cannot detect type of input layer "+ key0)
        else:
            w0_b = m0_state[key0_b].cpu().numpy()
            m0_state[key0_b] = torch.tensor(w0_b - norm_fact, device = orig_device)            
    m0_state[key0] = torch.tensor(w0, device = orig_device)
    
mean_rgb = {
        "": [0.0, 0.0, 0.0],
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
        "railsem19": [0.0, 0.0, 0.0],
        "vistas": [80.5423, 91.3162, 81.4312]}

def convert(args):
    m0 = torch.load(args.model_path)
    m0_state = m0["model_state"]
    norm_mean = [0.0,0.0,0.0]
    versions = [mean_rgb[v.strip()] for v in args.change_version.split(';')]
    if len(versions) == 2:
        norm_mean = [versions[0][c] - versions[1][c] for c in range(3)]
    normalization_factor = 1.0
    if not args.img_norm is None:
        if args.img_norm:
            normalization_factor = 255.0
        else:
            normalization_factor = 1.0/255.0
    inp_layers = [l.strip() for l in args.inp_layers.split(';')]
    if len(inp_layers) == 0 or len(inp_layers[0]) == 0:
        inp_layers = [list(m0_state.keys())[0]]
        if inp_layers[0] == "module.convbnrelu1_1.cbr_unit.0.weight":
            inp_layers.append("module.convbnrelu1_sub1.cbr_unit.0.weight")
    trg_path = args.out_path
    if len(trg_path) == 0:
        trg_path = args.model_path.replace('.pth','').replace('.pkl','')+'_transf.pkl'
    print("Model transformer applies these changes: normalization_shift, normalization_factor, flip_rgb", norm_mean, normalization_factor, args.flip_rgb)
    print("to these input layers: ", inp_layers)
    
    for l in inp_layers:
        if not l in m0_state:
            print("Warning: skipping unknown key "+l)
            continue
        transform_layer(m0_state, l, normalization_factor = normalization_factor, norm_mean = norm_mean)
    torch.save(m0, trg_path)
    
def main_convert(arg0):
    parser = argparse.ArgumentParser(description="Program to remove image pre-processor steps by applying them to model wheights directly.\nWARNING: this currently does not work for batch-norm models!\nParams:")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="frrnB_cityscapes_best_model_miou63.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--inp_layers", nargs="?", type=str, default="", help="Names of all input layers, default: use auto-detection"
    )
    parser.add_argument(
        "--change_version",
        nargs="?",
        type=str,
        default="",
        help="Change image mean normalization, command: <source_version>;<target_version>, e.g. cityscapes;pascal",
    )
    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Change image mean scaling (from [0;255] to [0;1])",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Change image mean scaling (from [0;1] to [0;255])",
    )
    parser.add_argument(
        "--flip_rgb",
        dest="no_img_norm",
        action="store_true",
        help="Flip input channels (rgb<->bgr)",
    )
    parser.add_argument(
        "--out_path", nargs="?", type=str, default="", help="Path for saving transformed model, default: inp + _transf.pkl"
    )
    parser.set_defaults(img_norm=None, flip_rgb=False)
    args = parser.parse_args(arg0)

    return convert(args)

if __name__ == "__main__":
    sys.exit(main_convert(sys.argv[1:]))
