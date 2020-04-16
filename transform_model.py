import os
import torch
from collections import OrderedDict
import argparse
import numpy as np
eps_bn = 1e-5 #default epsilon for bn

mean_rgb = {
        "": [0.0, 0.0, 0.0],
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
        "railsem19": [0.0, 0.0, 0.0],
        "vistas": [80.5423, 91.3162, 81.4312],
        "pascal_bgr": [123.68, 116.779, 103.939],
        "vistas_bgr": [81.4312, 91.3162, 80.5423]}

def transform_layer(m0_state, key0, normalization_factor = 1.0, norm_mean = [0.0,0.0,0.0], apply_bgr_flip = True):
    orig_device =  m0_state[key0].get_device()
    w0 = m0_state[key0].cpu().numpy().astype(np.float64)
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
            w0_b = m0_state[key0_b].cpu().numpy().astype(np.float64)
            m0_state[key0_b] = torch.tensor((w0_b - norm_fact).astype(np.float32), device = orig_device)            
    m0_state[key0] = torch.tensor(w0.astype(np.float32), device = orig_device)
    
def find_diffs_bn(state0, stateTempl):
    to_bn = {}
    from_bn = {}
    bn_vars = ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']
    for idx0, k in enumerate(stateTempl.keys()):
        if k in state0:
            continue
        k_split = k.split('.')
        if len(k_split) > 2 and k_split[-2] == '1' and k_split[-1] in bn_vars: #check if this is a bn node
            to_bn_name = k[:k.rfind('.')][:-2]+'.0'
            if to_bn_name+'.weight' in state0:
                if not to_bn_name in to_bn:
                    to_bn[to_bn_name] = (idx0, '.'.join(k_split[:-1])+'.bias' in stateTempl)
                continue
        if k.endswith('.0.bias'):
            from_bn_name = k[:k.rfind('.')][:-2]+'.1'
            if from_bn_name+'.running_mean' in state0:
                if not from_bn_name in from_bn:
                    from_bn[from_bn_name] = (idx0, None)
                continue
        print("Warning: template's key "+ k+" not found in loaded model (and not bn)")
    for idx0, k in enumerate(state0.keys()):
        if k in state0:
            continue
        to_bn_name = k[:k.rfind('.')]+'.0'
        if from_bnz in to_bn:
            continue
        from_bn_name = k[:k.rfind('.')]+'.1'
        if from_bn_name in from_bn:
            continue
        print("Warning: loaded model's key "+ k+" not found template (and not bn)")
    return to_bn, from_bn

def transform_from_bn(m0_state, key_from_bn):
    k0 = key_from_bn[:-2]+'.0.weight'
    k0bias = key_from_bn[:-2]+'.0.bias' #this entry should currently not exist!
    if not key_from_bn.endswith('.1') or not k0 in m0_state or \
       not key_from_bn+'.running_var' in m0_state or k0bias in m0_state:
        print("Warning: Skipping unknown batch entry "+k)
        return [],{}
        
    orig_device =  m0_state[k0].get_device()
    #bn: y = (x-running_mean)*gamma/sqrt(running_var+eps) + beta
    w1_var = m0_state[key_from_bn+'.running_var'].cpu().numpy().astype(np.float64)
    w1_var = 1.0/np.sqrt(w1_var+eps_bn)
    if key_from_bn+'.weight' in m0_state:
        w1_var = w1_var * m0_state[key_from_bn+'.weight'].cpu().numpy().astype(np.float64)
        
    w0_bias = -m0_state[key_from_bn+'.running_mean'].cpu().numpy().astype(np.float64) * w1_var
    if key_from_bn+'.bias' in m0_state:
        w0_bias += m0_state[key_from_bn+'.bias'].cpu().numpy().astype(np.float64)
    
    w0 = m0_state[k0].cpu().numpy().astype(np.float64)
    #apply batch norm weight accross output dim of previous node
    w0r = w0.reshape((w0.shape[0],-1))
    w0new = w0r*w1_var.reshape((w1_var.shape[0],1))
    w0new = w0new.reshape(w0.shape)
    
    m0_state[k0] = torch.tensor(np.copy(w0new).astype(np.float32), device = orig_device)
    remove_nodes = [key_from_bn+'.weight',key_from_bn+'.running_mean',
                    key_from_bn+'.running_var',key_from_bn+'.num_batches_tracked', key_from_bn+'.bias']
    append_nodes = {}
    append_nodes[k0] = (k0bias, torch.tensor(np.copy(w0_bias).astype(np.float32), device = orig_device)) # this bias term is added after the weights term
    return remove_nodes, append_nodes

def transform_to_bn(m0_state, key_to_bn, ref_is_affine):
    k0w = key_to_bn+'.weight'
    k1 = key_to_bn[:-2]+'.1'
    k1w = k1 + '.weight'
    k1bias = k1 +'.bias'
    k1runmean = k1 + '.running_mean'
    k1runvar = k1 + '.running_var'
    k1numbtracked = k1 + '.num_batches_tracked'
    if not key_to_bn.endswith('.0') or not k0w in m0_state or \
        k1+'.weight' in m0_state or k1+'.running_var' in m0_state or k1bias in m0_state:
        print("Warning: Cannot convert entry " + key_to_bn + " to bn")
        return [],{}
    append_nodes = {}
    orig_device =  m0_state[k0w].get_device()
    #bn: y = (x-running_mean)*gamma/sqrt(running_var+eps) + beta
    inp_dim = m0_state[k0w].shape[0]
    np.zeros((inp_dim,), dtype = np.float32)
    if ref_is_affine:
        append_nodes[k0w] = (k1w, torch.tensor(np.ones((inp_dim,), dtype = np.float32), device = orig_device))
        append_nodes[k1w] = (k1bias, torch.tensor(np.zeros((inp_dim,), dtype = np.float32), device = orig_device))
    else:
        k1bias = k0w # directly start with running_var
    if key_to_bn+'.bias' in m0_state:
        b0 = m0_state[key_to_bn+'.bias'].cpu().numpy().astype(np.float64)
        append_nodes[k1bias] = (k1runmean, torch.tensor((b0*-1.0).astype(np.float32), device = orig_device)) #use original bias running_mean; the other weights are set to identity
    else:
        append_nodes[k1bias] = (k1runmean, torch.tensor(np.zeros((inp_dim,), dtype = np.float32), device = orig_device)) # this bias term is added after the weights term
    append_nodes[k1runmean] = (k1runvar, torch.tensor(np.ones((inp_dim,), dtype = np.float32) - eps_bn, device = orig_device)) # this bias term is added after the weights term
    append_nodes[k1runvar] = (k1numbtracked, torch.tensor(np.zeros((inp_dim,), dtype = np.float32), device = orig_device)) # this bias term is added after the weights term
    remove_nodes = [key_to_bn+'.bias']
    return remove_nodes, append_nodes

def convert(args):
    m0 = torch.load(args.model_path)
    m0_state = m0["model_state"]
    norm_mean = [0.0,0.0,0.0]
    versions = [mean_rgb[v.strip()] for v in args.change_version.split(';')]
    if len(versions) == 2:
        norm_mean = [versions[1][c] - versions[0][c] for c in range(3)]
    if args.flip_rgb:
        norm_mean = norm_mean[::-1]
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
    
    num_templ = 0
    if len(args.target_template) > 0:
        
        #use template model file to identify differences resp. batch norm nodes
        m_trg_templ = torch.load(args.target_template)
        m_trg_templ_state = m_trg_templ["model_state"]
            
        to_bn, from_bn = find_diffs_bn(m0_state, m_trg_templ_state)

        remove_nodes = []
        append_nodes = {}

        for k, _ in from_bn.items():
            remove_nodes0, append_nodes0 = transform_from_bn(m0_state, k)
            remove_nodes += remove_nodes0
            append_nodes.update(append_nodes0)
        for k, (_, ref_is_affine) in to_bn.items():
            remove_nodes0, append_nodes0 = transform_to_bn(m0_state, k, ref_is_affine)
            remove_nodes += remove_nodes0
            append_nodes.update(append_nodes0)

        m1_state = OrderedDict()
        for k in m0_state:
            if k in remove_nodes:
                num_templ += 1
                continue
            m1_state[k] = m0_state[k]
            k_app = k
            while k_app in append_nodes:
                key_next, node0 = append_nodes.pop(k_app)
                k_app = key_next             
                m1_state[key_next] = node0
                num_templ += 1
        if len(append_nodes) > 0:
            kk = list(append_nodes.keys())
            print("Warning: Could not append %i nodes." % len(append_nodes), kk[0])
            
        m0["model_state"] = m1_state
        m0_state = m1_state
        
    print("Model transformer applies these changes: normalization_shift, normalization_factor, flip_rgb", norm_mean, normalization_factor, args.flip_rgb)
    print("to these input layers: ", inp_layers)
    if num_templ > 0:
        print("Changend %i nodes due to differences in template " % num_templ)
    
    for l in inp_layers:
        if not l in m0_state:
            print("Warning: skipping unknown key "+l)
            continue
        transform_layer(m0_state, l, normalization_factor = normalization_factor, norm_mean = norm_mean, apply_bgr_flip = args.flip_rgb)
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
        dest="flip_rgb",
        action="store_true",
        help="Flip input channels (rgb<->bgr)",
    )
    parser.add_argument(
        "--out_path", nargs="?", type=str, default="", help="Path for saving transformed model, default: inp + _transf.pkl"
    )
    parser.add_argument(
        "--target_template",
        nargs="?",
        type=str,
        default="",
        help="Use target model file to identify conversions between batch normalization nodes",
    )
    parser.set_defaults(img_norm=None, flip_rgb=False)
    args = parser.parse_args(arg0)

    return convert(args)

if __name__ == "__main__":
    sys.exit(main_convert(sys.argv[1:]))
