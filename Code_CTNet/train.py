import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.callbacks import LossHistory
from utils.dataloader import SiameseDataset, dataset_collate
from utils.utils import (download_weights, get_lr_scheduler, load_dataset,
                         set_optimizer_lr, show_config)
from utils.utils_fit import fit_one_epoch
from option import parser
from ACT import ACT

if __name__ == "__main__":
    # Parse command line arguments
    args = parser.parse_args()
    ngpus_per_node  = torch.cuda.device_count() # Get the number of GPUs available

    if args.distributed:
        # Initialize distributed training
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    # Download pretrained weights if required
    if args.pretrained:
        if args.distributed:
            if local_rank == 0:
                download_weights("vgg16")  
            dist.barrier()
        else:
            download_weights("vgg16")

    # Initialize model
    model = ACT(args)
    if args.model_path != '':
        #------------------------------------------------------#
        # Load weights if a model path is provided
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(args.model_path))

        model_dict      = args.model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        # Match pretrained weights with the model's weights
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        args.model.load_state_dict(model_dict)
        #------------------------------------------------------#
        # Display keys that were and were not loaded
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mKind reminder: It is normal if the head part is not loaded, but it is an error if the backbone part is not loaded.\033[0m")
    
    #----------------------#
    # Define loss function
    #----------------------#
    loss = nn.BCEWithLogitsLoss()
    #----------------------#
    # Initialize LossHistory for logging
    #----------------------#
    if local_rank == 0:
        loss_history = LossHistory(args.save_dir, model, input_shape=args.input_shape)
    else:
        loss_history = None
        
    #------------------------------------------------------------------#
    # Initialize GradScaler for mixed precision training if fp16 is enabled
    #------------------------------------------------------------------#
    if args.fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model
    #----------------------------#
    # Convert model to SyncBatchNorm if distributed training with multiple GPUs
    #----------------------------#
    if args.sync_bn and ngpus_per_node > 1 and args.distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif args.sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    # Move model to CUDA device
    if args.Cuda:
        if args.distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = model_train.to(device = torch.device('cuda:0'))

    #----------------------------------------------------#
    # Load dataset
    #----------------------------------------------------#
    train_ratio = 0.9
    train_lines, train_labels, val_lines, val_labels = load_dataset(args.dataset_path, args.train_own_data, train_ratio)
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            model_path = args.model_path, input_shape = args.input_shape, \
            Init_Epoch = args.Init_Epoch, Epoch = args.Epoch, batch_size = args.batch_size, \
            Init_lr = args.Init_lr, Min_lr = args.Init_lr*0.01, optimizer_type = args.optimizer_type, momentum = args.momentum, lr_decay_type = args.lr_decay_type, \
            save_period = args.save_period, save_dir = args.save_dir, num_workers = args.num_workers, num_train = num_train, num_val = num_val
        )

        # Calculate total steps and display warnings if steps are less than recommended
        wanted_step = 3e4 if args.optimizer_type == "sgd" else 1e4
        total_step  = num_train // args.batch_size * args.Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // args.batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using %s optimizer, it is recommended to set the total training steps to above %d.\033[0m"%(args.optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] The total training data this run is %d, batch_size is %d, a total of %d Epochs, calculated total training steps are %d.\033[0m"%(num_train, args.batch_size, args.Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total training steps are %d, less than the recommended total steps %d, it is recommended to set the total epochs to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))


    if True:
        #-------------------------------------------------------------------#
        # Adjust learning rate based on batch size
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if args.optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if args.optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(args.batch_size / nbs * args.Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(args.batch_size / nbs * args.Init_lr*0.01, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        # Select optimizer
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (args.momentum, 0.999), weight_decay = args.weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=args.momentum, nesterov=True, weight_decay = args.weight_decay)
        }[args.optimizer_type]

        #---------------------------------------#
        # Get learning rate scheduler
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.Epoch)
        
        #---------------------------------------#
        # Determine the number of steps per epoch
        #---------------------------------------#
        epoch_step      = num_train // args.batch_size
        epoch_step_val  = num_val // args.batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset too small, cannot continue training, please expand the dataset.")

        train_dataset   = SiameseDataset(args.input_shape, train_lines, train_labels, True)
        val_dataset     = SiameseDataset(args.input_shape, val_lines, val_labels, False)

        if args.distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = args.batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        # Create DataLoaders for training and validation
        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)

        for epoch in range(args.Init_Epoch, args.Epoch):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # Train for one epoch
            fit_one_epoch(model_train, model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, args.Epoch, args.Cuda, args.fp16, scaler, args.save_period, args.save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()

