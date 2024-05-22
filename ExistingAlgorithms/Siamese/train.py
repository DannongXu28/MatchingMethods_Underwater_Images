import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy

from nets.siamese import Siamese
from utils.callbacks import LossHistory
from utils.dataloader import SiameseDataset, dataset_collate
from utils.utils import download_weights, get_lr_scheduler, load_dataset, set_optimizer_lr, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # Configuration settings
    Cuda = True
    distributed = False
    sync_bn = False
    fp16 = False
    dataset_path = "datasets"
    input_shape = [105, 105]
    train_own_data = True
    pretrained = False
    model_path = ""

    Init_Epoch = 0
    Epoch = 100
    batch_size = 32

    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 5e-4
    lr_decay_type = 'cos'
    save_period = 10
    save_dir = 'logs'
    num_workers = 4

    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # Download pretrained weights if needed
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights("vgg16")
            dist.barrier()
        else:
            download_weights("vgg16")

    # Initialize the model
    model = Siamese(input_shape, pretrained)

    # Load model weights if a path is provided
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mKind reminder: It is normal for the head part to not be loaded, but it is an error if the Backbone part is not loaded.\033[0m")

    # Define the loss function
    loss = nn.BCEWithLogitsLoss()

    # Initialize loss history for logging
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # Initialize mixed precision scaler if fp16 is enabled
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    # Set model to training mode
    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    # Move model to GPU(s) if CUDA is available
    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # Use BinaryAccuracy instead of MulticlassAccuracy
    train_accuracy_metric = BinaryAccuracy()
    val_accuracy_metric = BinaryAccuracy()

    # Load the dataset
    train_ratio = 0.9
    train_lines, train_labels, val_lines, val_labels = load_dataset(dataset_path, train_own_data, train_ratio)
    num_train = len(train_lines)
    num_val = len(val_lines)

    # Display configuration if on the main process
    if local_rank == 0:
        show_config(
            model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        wanted_step = 3e4 if optimizer_type == "sgd" else 1e4
        total_step = num_train // batch_size * Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training steps to above %d.\033[0m" % (optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] For this run, the total training data volume is %d, with a batch size of %d, training for %d epochs, resulting in a total training step count of %d.\033[0m" % (num_train, batch_size, Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total training steps are %d, which is less than the recommended total steps of %d, it is suggested to set the total epochs to %d.\033[0m" % (total_step, wanted_step, wanted_epoch))

    # Set up learning rate limits and scheduler
    if True:
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training. Please expand the dataset.")

        # Create datasets and data loaders
        train_dataset = SiameseDataset(input_shape, train_lines, train_labels, True)
        val_dataset = SiameseDataset(input_shape, val_lines, val_labels, False)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)

        # Training loop
        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            train_loss, val_loss, train_accuracy, val_accuracy = fit_one_epoch(
                model_train, model, loss, loss_history, optimizer, epoch, epoch_step,
                epoch_step_val, gen, gen_val, Epoch, Cuda, fp16, scaler, save_period,
                save_dir, local_rank
            )

            if local_rank == 0:
                loss_history.append_metrics(epoch, train_loss, val_loss, train_accuracy, val_accuracy)

