import os,sys,math
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np

from torch.utils import data
from tqdm import tqdm_notebook as tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter
import resource
import psutil
import gc

def report_cuda_mem(device0=None):
    bytetogb = 1.0/float(1024*1024*1024)
    return "A: %0.3f/%0.3f R: %0.3f/%0.3f" % (float(torch.cuda.memory_allocated(device=device0))*bytetogb, float(torch.cuda.max_memory_allocated(device=device0))*bytetogb,float(torch.cuda.memory_cached(device=device0))*bytetogb, float(torch.cuda.max_memory_cached(device=device0))*bytetogb)

def report_mem_both(device0=None):
    #not found resource.RUSAGE_BOTH
    kbtogb = 1.0/float(1024*1024)
    bytetogb = 1.0/float(1024*1024*1024)
    cpu_usage_self = resource.getrusage(resource.RUSAGE_SELF) 
    cpu_usage_child = resource.getrusage(resource.RUSAGE_CHILDREN) 
    cpu_usage_curr = float(psutil.Process(os.getpid()).memory_info().rss)
    ret_str = "CPU:  %0.3f/%0.3f " % (cpu_usage_curr*bytetogb, float(cpu_usage_self.ru_maxrss + cpu_usage_child.ru_maxrss)*kbtogb)
    ret_str += " GPU "+report_cuda_mem(device0)
    return ret_str

def calc_histogram(cfg, split_name):
    data_path = cfg["data"]["path"]
    data_loader = get_loader(cfg["data"]["dataset"])
    h_loader = data_loader(
        data_path,
        is_transform=True,
        split=split_name,
        img_size=(("same", "same")),
        version=cfg["data"].get("version","cityscapes"),  
        img_norm=False,
        offline_res="hist"
    )
    hist_loader = data.DataLoader( h_loader, batch_size=1, num_workers=cfg["training"]["n_workers"] )
    max_num = len(hist_loader)
    idx_to_name = {}
    name_to_idx = {}
    max_label_num = h_loader.n_classes + 1
    hist_all = np.zeros((max_num, max_label_num), dtype=np.uint64)
    for i_val, (images_vals, labels_hist) in tqdm(enumerate(hist_loader), total = max_num, desc = "Calculating "+split_name+" histograms"):
        label_hist_np = labels_hist.numpy()
        idx_to_name[i_val] = images_vals
        hist_all[i_val,0:label_hist_np.shape[1]] = label_hist_np[0,:]
    return hist_all, idx_to_name

def hist_to_framenames(hist_all, idx_to_name, boost_idx, boost_max_ratio):
    if not isinstance(boost_idx, list):
        boost_idx = [boost_idx]
    ids_val = []
    ids_covered = {}
    for idx0 in boost_idx:
        if idx0 < 0:
            continue
        ids_val0 = list(np.nonzero(hist_all[:,idx0])[0])
        #print("Adding frames for %i (max new num: %i, old num: %i)" % (idx0, len(ids_val0), len(ids_val)))
        for add0 in ids_val0:
            first_id = ids_covered.get(add0,idx0)
            ids_val.append((add0, first_id))
            ids_covered[add0] = first_id
    num_total = int(float(len(hist_all)) * boost_max_ratio+0.5)
    if num_total > len(ids_val):
        ids_containing = set(ids_covered.keys())
        ids_remainder = list(set(range(hist_all.shape[0]))-ids_containing)
        num_add = min(num_total - len(ids_val), len(ids_remainder))
        if num_add > 0:
            add_ids = np.random.choice(len(ids_remainder), num_add)
            for id0 in add_ids:
                ids_val.append((ids_remainder[id0],-1))
    all_frames = []
    for id0, bidx0 in ids_val:
        all_frames.append((idx_to_name[id0][0],bidx0))
    return all_frames

def train(cfg, writer, logger):
    bytetogb = 1.0/float(1024*1024*1024)
    cuda_is_available = torch.cuda.is_available()
    report_mem = True #cfg.get("report_mem",False)
    calc_loss_on_cpu = False#True #cuda_is_available and cfg.get("calc_loss_on_cpu",False) 
    
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))
    
    if calc_loss_on_cpu:
        device_cpu = torch.device("cpu")
    
     # Setup device
    device = torch.device("cuda" if cuda_is_available else "cpu")
    if report_mem:
        print("0.) After device setup (total free: %0.3f): " % (bytetogb * float(torch.cuda.get_device_properties(device).total_memory)) + report_mem_both())

    # Special index boosting data:
    boost_indices = cfg["training"].get("boost_indices", -1)
    if not isinstance(boost_indices, list):
        boost_indices = [boost_indices]
    boost_max_ratio = cfg["training"].get("boost_max_ratio", -1)
    val_frame_list = None
    if len(boost_indices) > 0:
        hist_val, idx_to_name_val = calc_histogram(cfg, cfg["data"]["val_split"])
        hist_train, idx_to_name_train = calc_histogram(cfg, cfg["data"]["train_split"])
        
    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)
    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]
    val_delta = cfg["data"].get("val_asp_ratio_delta", -1.0)
    if val_delta < 1.0:
        val_delta = -1.0

    if len(boost_indices) == 0:
        t_loader = data_loader(
            data_path,
            is_transform=True,
            split=cfg["data"]["train_split"],
            img_size=(cfg["data"].get("img_rows","same"), cfg["data"].get("img_cols", "same")),
            version=cfg["data"].get("version","cityscapes"),
            img_norm=cfg["data"].get("img_norm",True),
            augmentations=data_aug,
        )
    
    val_augmentations = None
    val_retries = 1
    if "validation" in cfg and "augmentations" in cfg["validation"]:
        val_augmentations =  get_composed_augmentations(cfg["validation"]["augmentations"])
        val_retries = cfg["validation"].get("boost_retries",1)
        
    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(cfg["model"].get("input_size",[cfg["data"].get("img_rows","same"), "same"])[0] , cfg["model"].get("input_size",["same",cfg["data"].get("img_cols", "same")])[1]),
        version=cfg["data"].get("version","cityscapes"),
        asp_ratio_delta_min = cfg["data"].get("val_asp_ratio_delta_min", 1.0/val_delta),
        asp_ratio_delta_max = cfg["data"].get("val_asp_ratio_delta_max", val_delta),      
        img_norm=cfg["data"].get("img_norm",True),
        augmentations=val_augmentations,
        frame_list = None
    )

    n_classes = v_loader.n_classes
    
    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = get_model(cfg["model"], n_classes).to(device)
    
    if report_mem:
        print("7.)After model loader setup: "+report_mem_both(device))
    
    for param in model.parameters():
        param.requires_grad = True
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
   
    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg) 
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))
    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])
    
    loss_fn = get_loss_function(cfg) 
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer1 from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            
            if report_mem:
                print("12.)After checkpnt loading: "+report_mem_both(device))

            model.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint and not cfg["training"].get("reset_optimizer", False):
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                scheduler.load_state_dict(checkpoint["scheduler_state"])
                start_iter = checkpoint["epoch"]
            else:
                logger.info("Resetting optimizer/scheduler")
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], start_iter
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True
    max_iters = cfg["training"]["train_iters"]
    boost_cnt = 0
    while i <= max_iters and flag:
        if 'reset_epoch' in cfg["training"]:
            scheduler.last_epoch = cfg["training"]['reset_epoch']
        train_frame_list = None
        if len(boost_indices) > 0:
            curr_boost_idx = boost_indices[boost_cnt % len(boost_indices)]
            boost_cnt += 1
            if not isinstance(curr_boost_idx, list):
                curr_boost_idx = [curr_boost_idx]
            if curr_boost_idx[0] < 0:
                train_frame_list = None
            else:
                logger.info(" Trying boosting for indices "+str(curr_boost_idx))
                train_frame_list = hist_to_framenames(hist_train, idx_to_name_train, curr_boost_idx, boost_max_ratio)
            t_loader = data_loader(
                data_path,
                is_transform=True,
                split=cfg["data"]["train_split"],
                img_size=(cfg["data"].get("img_rows","same"), cfg["data"].get("img_cols", "same")),
                version=cfg["data"].get("version","cityscapes"),
                img_norm=cfg["data"].get("img_norm",True),
                augmentations=data_aug,
                frame_list=train_frame_list,
                boost_retries = cfg["training"].get("boost_retries",1)
            )
            if curr_boost_idx[0] < 0:
                val_frame_list = None
            else:
                val_frame_list = hist_to_framenames(hist_val, idx_to_name_val, curr_boost_idx, boost_max_ratio)
            v_loader = data_loader(
                data_path,
                is_transform=True,
                split=cfg["data"]["val_split"],
                img_size=(cfg["model"].get("input_size",[cfg["data"].get("img_rows","same"), "same"])[0] , cfg["model"].get("input_size",["same",cfg["data"].get("img_cols", "same")])[1]),
                version=cfg["data"].get("version","cityscapes"),
                asp_ratio_delta_min = cfg["data"].get("val_asp_ratio_delta_min", 1.0/val_delta),
                asp_ratio_delta_max = cfg["data"].get("val_asp_ratio_delta_max", val_delta),      
                img_norm=cfg["data"].get("img_norm",True),
                augmentations=val_augmentations,
                boost_retries = val_retries,
                frame_list = val_frame_list
            )

        #reshuffle training set with each epoch
        trainloader = data.DataLoader(
           t_loader,
           batch_size=cfg["training"]["batch_size"],
           num_workers=cfg["training"]["n_workers"],
           shuffle=True,
        )
        
        num_elems = len(trainloader)
        training_iters = cfg["training"]["val_interval"]
        if abs(num_elems-training_iters) < 25:
            training_iters = num_elems
        i += int(math.ceil(float(i)/float(training_iters))-i) # get to next epoch start
        printing_iters = cfg["training"].get("print_interval",cfg["training"]["val_interval"])
        if abs(cfg["training"]["val_interval"]-cfg["training"]["print_interval"]) < 25:
            printing_iters = training_iters

        t_max, i_start = len(trainloader), i
        for (images, labels) in tqdm(trainloader, desc = 'Training Epoch %i; mIouMax: %f'%(int(i//training_iters),best_iou)):
            i += 1
            
            start_ts = time.time()
            #optimizer.zero_grad()
            #optimizer.step()
            scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            if i == i_start+2 and report_mem: #c1
                print("19.)After device copy: "+report_mem_both(device))
            
            optimizer.zero_grad()
            outputs = model(images)
            if i == i_start+2 and report_mem: #c2
                print("21.)After training step: "+report_mem_both(device))

            loss = loss_fn(input=outputs, target=labels)

            if i == i_start+2 and report_mem: #c3
                print("22.)Before loss step: "+report_mem_both(device))
            
            loss.backward()
            if i == i_start+2 and report_mem: #c4
                print("23.)After loss step: "+report_mem_both(device))
            optimizer.step()
            time_meter.update(time.time() - start_ts)

            if (i + 1) % printing_iters == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    time_meter.avg / cfg["training"]["batch_size"],
                )

                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if i  % training_iters == 0 or (i + 1 - i_start) == t_max:
            #if i > i_start + 2:
                model.eval()
                with torch.no_grad():
                    valloader = data.DataLoader( v_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"] )
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader), desc = "Validation"):
                        if i_val < 1 and report_mem:
                            print("24)Before val device copy: "+report_mem_both(device))
                        
                        images_val = images_val.to(device)
                        if i_val < 1 and report_mem:
                            print("25.)After val device copy: "+report_mem_both(device))
            
                        outputs = model(images_val)
                        del images_val
                        if i_val < 1 and report_mem:
                            print("26.)After val prediction: "+report_mem_both(device))
                        
                        pred = outputs.data.max(1)[1].cpu().numpy()
                        if calc_loss_on_cpu:
                            outputs_cpu = outputs.data.cpu()
                            del outputs
                            labels_val = labels_val.to(device_cpu)
                            if i_val < 1 and report_mem:
                                print("26.5)After labelsval to device: "+report_mem_both(device))
                            loss_fn_cpu = get_loss_function(cfg)
                            if i_val < 1 and report_mem:
                                print("26.5)After loss_fn gotten: "+report_mem_both(device))
                            val_loss = loss_fn_cpu(input=outputs_cpu, target=labels_val)
                            if i_val < 1 and report_mem:
                                print("26.5)After loss_fn_cpu prediction: "+report_mem_both(device))
                            del outputs_cpu
                            gt = labels_val.numpy()
                            del labels_val
                            del loss_fn_cpu
                            gc.collect()
                        else:
                            labels_val = labels_val.to(device)
                            val_loss = loss_fn(input=outputs, target=labels_val)
                            gt = labels_val.data.cpu().numpy()
                        if i_val < 1 and report_mem:
                            print("27.)After val loss calc: "+report_mem_both(device))
            
                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())
                        del val_loss
                        #break
                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

                if report_mem:
                    print("30.) After running metrics val: "+report_mem_both(device))
            
                val_loss_meter.reset()
                running_metrics_val.reset()
                
                is_best_iou = score["Mean IoU : \t"] >= best_iou
                name_postfix = "_last_epoche"
                if is_best_iou:
                    best_iou = score["Mean IoU : \t"]
                    name_postfix = "_best_model"
                if is_best_iou or len(boost_indices) > 0:
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}{}.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"],name_postfix),
                    )
                    torch.save(state, save_path)
                break

            #if i > i_start + 2:
            if (i + 1) == max_iters:
                flag = False
                break


if __name__ == "__main__":
    sys.exit(main_tr())

def main_tr(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args(argv)

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    run_id = random.randint(1, 100000)
    logdir = os.path.join("/workspace/data/pytsegmruns", os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    train(cfg, writer, logger)
