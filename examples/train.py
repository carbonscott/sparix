import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import random
import tqdm
import logging
import math
import h5py
import signal
import sys

from sparix.data                 import FrameDataset
from sparix.modeling.transformer import Transformer
from sparix.utils                import init_logger, MetaLog, save_checkpoint, load_checkpoint, set_seed
from sparix.trans import Pad

torch.autograd.set_detect_anomaly(True)

seed = 0
set_seed(seed)

logger = logging.getLogger(__name__)

# [[[ USER INPUT ]]]
timestamp_prev = None # "2023_0505_1249_26"
epoch          = None # 21

drc_chkpt = "chkpts"
fl_chkpt_prev   = None if timestamp_prev is None else f"{timestamp_prev}.epoch_{epoch}.chkpt"
path_chkpt_prev = None if fl_chkpt_prev is None else os.path.join(drc_chkpt, fl_chkpt_prev)

timestamp = init_logger(returns_timestamp = True)

path_h5 = "3IYF.Fibonacci.h5"
fh = h5py.File(path_h5, 'r')

# Set up signal to detect Ctrl-C to close h5 file...
def close_h5_on_ctrl_c(fh_h5):
    def signal_handler(signal_received, stack_frame):
        print('SIGINT or CTRL-C detected...')
        print(f'...Closing h5')
        fh_h5.close()
        print('...h5 is closed.')
        sys.exit(0)

    return signal_handler

handler = close_h5_on_ctrl_c(fh)
signal.signal(signal.SIGINT, handler)

# Fetch the data source...
frame_source = fh["intensities"]
frame_source.shape

# Figure out the padding
N, C, H, W = frame_source.shape
H_patch = 16
W_patch = 16
H_patch_count = math.ceil(H / H_patch)
W_patch_count = math.ceil(W / W_patch)
H_padded = H_patch_count * H_patch
W_padded = W_patch_count * W_patch
pad = Pad(H_padded, W_padded)

train_frac              = 0.7
train_frame_idx_list    = range(N)[:int(train_frac * N) ]
validate_frame_idx_list = range(N)[ int(train_frac * N):]

num_frame_in_context = 4
batch_size           = int(2e2)
sample_size          = int(1e3)
num_workers          = 16

dataset_train    = FrameDataset(frame_source, train_frame_idx_list,    num_frame_in_context, H_patch, W_patch, sample_size, pad)
dataset_validate = FrameDataset(frame_source, validate_frame_idx_list, num_frame_in_context, H_patch, W_patch, sample_size, pad)

dataloader_train = torch.utils.data.DataLoader( dataset_train,
                                                shuffle     = True,
                                                pin_memory  = True,
                                                batch_size  = batch_size,
                                                num_workers = num_workers, )
dataloader_validate = torch.utils.data.DataLoader( dataset_validate,
                                                shuffle     = True,
                                                pin_memory  = True,
                                                batch_size  = batch_size,
                                                num_workers = num_workers, )

# Define model...
tok_size            = H_patch * W_patch
embd_size           = 768    # For using a pre-trained model from google
context_length      = num_frame_in_context * H_patch_count * W_patch_count
num_blocks          = 4
num_heads           = 4
uses_causal_mask    = True
attention_dropout   = 0.1
residual_dropout    = 0.1
feedforward_dropout = 0.1

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model  = Transformer(tok_size            = tok_size,
                     embd_size           = embd_size,
                     context_length      = context_length,
                     num_blocks          = num_blocks,
                     num_heads           = num_heads,
                     uses_causal_mask    = uses_causal_mask,
                     attention_dropout   = attention_dropout,
                     residual_dropout    = residual_dropout,
                     feedforward_dropout = feedforward_dropout,)
model.to(device)
logger.info(f'{sum(p.numel() for p in model.parameters())/1e6}, M parameters')

loads_pretrained_model = True
if loads_pretrained_model:
    logger.info(f'Loading pretrained weights...')
    path_model_weight = "chkpts/google.vit-base-patch16-224-in21k"
    model_weight = torch.load(path_model_weight)
    model_weight_dict = model_weight.get('model_state_dict')

    google_to_custom_dict = {
        "transformer_block.0.multi_head_att_layer.proj_q.weight"      : "vit.encoder.layer.0.attention.attention.query.weight",
        "transformer_block.0.multi_head_att_layer.proj_q.bias"        : "vit.encoder.layer.0.attention.attention.query.bias",
        "transformer_block.0.multi_head_att_layer.proj_k.weight"      : "vit.encoder.layer.0.attention.attention.key.weight",
        "transformer_block.0.multi_head_att_layer.proj_k.bias"        : "vit.encoder.layer.0.attention.attention.key.bias",
        "transformer_block.0.multi_head_att_layer.proj_v.weight"      : "vit.encoder.layer.0.attention.attention.value.weight",
        "transformer_block.0.multi_head_att_layer.proj_v.bias"        : "vit.encoder.layer.0.attention.attention.value.bias",
        "transformer_block.0.multi_head_att_layer.proj_linear.weight" : "vit.encoder.layer.0.attention.output.dense.weight",
        "transformer_block.0.multi_head_att_layer.proj_linear.bias"   : "vit.encoder.layer.0.attention.output.dense.bias",
        "transformer_block.0.ff_layer.ff_layer.0.weight"              : "vit.encoder.layer.0.intermediate.dense.weight",
        "transformer_block.0.ff_layer.ff_layer.0.bias"                : "vit.encoder.layer.0.intermediate.dense.bias",
        "transformer_block.0.ff_layer.ff_layer.2.weight"              : "vit.encoder.layer.0.output.dense.weight",
        "transformer_block.0.ff_layer.ff_layer.2.bias"                : "vit.encoder.layer.0.output.dense.bias",
        "transformer_block.0.layer_norm_pre_multi_head.weight"        : "vit.encoder.layer.0.layernorm_before.weight",
        "transformer_block.0.layer_norm_pre_multi_head.bias"          : "vit.encoder.layer.0.layernorm_before.bias",
        "transformer_block.0.layer_norm_pre_feedforward.weight"       : "vit.encoder.layer.0.layernorm_after.weight",
        "transformer_block.0.layer_norm_pre_feedforward.bias"         : "vit.encoder.layer.0.layernorm_after.bias",
        "transformer_block.1.multi_head_att_layer.proj_q.weight"      : "vit.encoder.layer.1.attention.attention.query.weight",
        "transformer_block.1.multi_head_att_layer.proj_q.bias"        : "vit.encoder.layer.1.attention.attention.query.bias",
        "transformer_block.1.multi_head_att_layer.proj_k.weight"      : "vit.encoder.layer.1.attention.attention.key.weight",
        "transformer_block.1.multi_head_att_layer.proj_k.bias"        : "vit.encoder.layer.1.attention.attention.key.bias",
        "transformer_block.1.multi_head_att_layer.proj_v.weight"      : "vit.encoder.layer.1.attention.attention.value.weight",
        "transformer_block.1.multi_head_att_layer.proj_v.bias"        : "vit.encoder.layer.1.attention.attention.value.bias",
        "transformer_block.1.multi_head_att_layer.proj_linear.weight" : "vit.encoder.layer.1.attention.output.dense.weight",
        "transformer_block.1.multi_head_att_layer.proj_linear.bias"   : "vit.encoder.layer.1.attention.output.dense.bias",
        "transformer_block.1.ff_layer.ff_layer.0.weight"              : "vit.encoder.layer.1.intermediate.dense.weight",
        "transformer_block.1.ff_layer.ff_layer.0.bias"                : "vit.encoder.layer.1.intermediate.dense.bias",
        "transformer_block.1.ff_layer.ff_layer.2.weight"              : "vit.encoder.layer.1.output.dense.weight",
        "transformer_block.1.ff_layer.ff_layer.2.bias"                : "vit.encoder.layer.1.output.dense.bias",
        "transformer_block.1.layer_norm_pre_multi_head.weight"        : "vit.encoder.layer.1.layernorm_before.weight",
        "transformer_block.1.layer_norm_pre_multi_head.bias"          : "vit.encoder.layer.1.layernorm_before.bias",
        "transformer_block.1.layer_norm_pre_feedforward.weight"       : "vit.encoder.layer.1.layernorm_after.weight",
        "transformer_block.1.layer_norm_pre_feedforward.bias"         : "vit.encoder.layer.1.layernorm_after.bias",
        "transformer_block.2.multi_head_att_layer.proj_q.weight"      : "vit.encoder.layer.2.attention.attention.query.weight",
        "transformer_block.2.multi_head_att_layer.proj_q.bias"        : "vit.encoder.layer.2.attention.attention.query.bias",
        "transformer_block.2.multi_head_att_layer.proj_k.weight"      : "vit.encoder.layer.2.attention.attention.key.weight",
        "transformer_block.2.multi_head_att_layer.proj_k.bias"        : "vit.encoder.layer.2.attention.attention.key.bias",
        "transformer_block.2.multi_head_att_layer.proj_v.weight"      : "vit.encoder.layer.2.attention.attention.value.weight",
        "transformer_block.2.multi_head_att_layer.proj_v.bias"        : "vit.encoder.layer.2.attention.attention.value.bias",
        "transformer_block.2.multi_head_att_layer.proj_linear.weight" : "vit.encoder.layer.2.attention.output.dense.weight",
        "transformer_block.2.multi_head_att_layer.proj_linear.bias"   : "vit.encoder.layer.2.attention.output.dense.bias",
        "transformer_block.2.ff_layer.ff_layer.0.weight"              : "vit.encoder.layer.2.intermediate.dense.weight",
        "transformer_block.2.ff_layer.ff_layer.0.bias"                : "vit.encoder.layer.2.intermediate.dense.bias",
        "transformer_block.2.ff_layer.ff_layer.2.weight"              : "vit.encoder.layer.2.output.dense.weight",
        "transformer_block.2.ff_layer.ff_layer.2.bias"                : "vit.encoder.layer.2.output.dense.bias",
        "transformer_block.2.layer_norm_pre_multi_head.weight"        : "vit.encoder.layer.2.layernorm_before.weight",
        "transformer_block.2.layer_norm_pre_multi_head.bias"          : "vit.encoder.layer.2.layernorm_before.bias",
        "transformer_block.2.layer_norm_pre_feedforward.weight"       : "vit.encoder.layer.2.layernorm_after.weight",
        "transformer_block.2.layer_norm_pre_feedforward.bias"         : "vit.encoder.layer.2.layernorm_after.bias",
        "transformer_block.3.multi_head_att_layer.proj_q.weight"      : "vit.encoder.layer.3.attention.attention.query.weight",
        "transformer_block.3.multi_head_att_layer.proj_q.bias"        : "vit.encoder.layer.3.attention.attention.query.bias",
        "transformer_block.3.multi_head_att_layer.proj_k.weight"      : "vit.encoder.layer.3.attention.attention.key.weight",
        "transformer_block.3.multi_head_att_layer.proj_k.bias"        : "vit.encoder.layer.3.attention.attention.key.bias",
        "transformer_block.3.multi_head_att_layer.proj_v.weight"      : "vit.encoder.layer.3.attention.attention.value.weight",
        "transformer_block.3.multi_head_att_layer.proj_v.bias"        : "vit.encoder.layer.3.attention.attention.value.bias",
        "transformer_block.3.multi_head_att_layer.proj_linear.weight" : "vit.encoder.layer.3.attention.output.dense.weight",
        "transformer_block.3.multi_head_att_layer.proj_linear.bias"   : "vit.encoder.layer.3.attention.output.dense.bias",
        "transformer_block.3.ff_layer.ff_layer.0.weight"              : "vit.encoder.layer.3.intermediate.dense.weight",
        "transformer_block.3.ff_layer.ff_layer.0.bias"                : "vit.encoder.layer.3.intermediate.dense.bias",
        "transformer_block.3.ff_layer.ff_layer.2.weight"              : "vit.encoder.layer.3.output.dense.weight",
        "transformer_block.3.ff_layer.ff_layer.2.bias"                : "vit.encoder.layer.3.output.dense.bias",
        "transformer_block.3.layer_norm_pre_multi_head.weight"        : "vit.encoder.layer.3.layernorm_before.weight",
        "transformer_block.3.layer_norm_pre_multi_head.bias"          : "vit.encoder.layer.3.layernorm_before.bias",
        "transformer_block.3.layer_norm_pre_feedforward.weight"       : "vit.encoder.layer.3.layernorm_after.weight",
        "transformer_block.3.layer_norm_pre_feedforward.bias"         : "vit.encoder.layer.3.layernorm_after.bias",
    }
    encoder_state_dict = model.state_dict()
    for k in encoder_state_dict.keys():
        if not k in google_to_custom_dict: continue
        k_google = google_to_custom_dict[k]
        encoder_state_dict[k] = model_weight_dict[k_google]
    model.load_state_dict(encoder_state_dict)

criterion = nn.MSELoss()

lr = 1e-3
weight_decay = 1e-4
param_iter = model.module.parameters() if hasattr(model, "module") else model.parameters()
optimizer = optim.AdamW(param_iter,
                        lr = lr,
                        weight_decay = weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode           = 'min',
                                         factor         = 2e-1,
                                         patience       = 20,
                                         threshold      = 1e-4,
                                         threshold_mode ='rel',
                                         verbose        = True)
## scheduler = None


# [[[ TRAIN LOOP ]]]
max_epochs = 5000

# From a prev training???
epoch_min = 0
loss_min  = float('inf')
if path_chkpt_prev is not None:
    epoch_min, loss_min = load_checkpoint(model, optimizer, scheduler, path_chkpt_prev)
    ## epoch_min, loss_min = load_checkpoint(model, None, None, path_chkpt_prev)
    epoch_min += 1    # Next epoch
    logger.info(f"PREV - epoch_min = {epoch_min}, loss_min = {loss_min}")

logger.info(f"Current timestamp: {timestamp}")

uses_mixed_precision = True
chkpt_saving_period  = 10
epoch_unstable_end  = 200
for epoch in tqdm.tqdm(range(max_epochs)):
    epoch += epoch_min

    # Uses mixed precision???
    if uses_mixed_precision: scaler = torch.cuda.amp.GradScaler()

    # ___/ TRAIN \___
    # Turn on training related components in the model...
    model.train()

    # Fetch batches...
    train_loss_list = []
    batch_train = tqdm.tqdm(enumerate(dataloader_train), total = len(dataloader_train), disable=True)
    for batch_idx, batch_entry in batch_train:
        # Unpack the batch entry and move them to device...
        batch_input, batch_target = batch_entry    # (B, T, H, W)

        batch_input  =  batch_input.to(device).view(batch_size, context_length,-1)
        batch_target = batch_target.to(device).view(batch_size, context_length,-1)

        # Forward, backward and update...
        if uses_mixed_precision:
            with torch.cuda.amp.autocast(dtype = torch.float16):
                # Forward pass...
                batch_output = model(batch_input)

                # Calculate the loss...
                loss = criterion(batch_output, batch_target)
                loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

            # Backward pass and optimization...
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass...
            batch_output = model(batch_input)

            # Calculate the loss...
            loss = criterion(batch_output, batch_target)
            loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

            # Backward pass and optimization...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Reporting...
        train_loss_list.append(loss.item())

    train_loss_mean = torch.mean(torch.tensor(train_loss_list))
    logger.info(f"MSG (device:{device}) - epoch {epoch}, mean train loss = {train_loss_mean:.8f}")


    # ___/ VALIDATE \___
    model.eval()

    # Fetch batches...
    validate_loss_list = []
    batch_validate = tqdm.tqdm(enumerate(dataloader_validate), total = len(dataloader_validate), disable = True)
    for batch_idx, batch_entry in batch_validate:
        # Unpack the batch entry and move them to device...
        batch_input, batch_target = batch_entry

        batch_input  =  batch_input.to(device).view(batch_size, context_length,-1)
        batch_target = batch_target.to(device).view(batch_size, context_length,-1)

        # Forward only...
        with torch.no_grad():
            if uses_mixed_precision:
                with torch.cuda.amp.autocast(dtype = torch.float16):
                    # Forward pass...
                    batch_output = model(batch_input)

                    # Calculate the loss...
                    loss = criterion(batch_output, batch_target)
                    loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus
            else:
                # Forward pass...
                batch_output = model(batch_input)

                # Calculate the loss...
                loss = criterion(batch_output, batch_target)
                loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

        # Reporting...
        validate_loss_list.append(loss.item())

    validate_loss_mean = torch.mean(torch.tensor(validate_loss_list))
    logger.info(f"MSG (device:{device}) - epoch {epoch}, mean val   loss = {validate_loss_mean:.8f}")

    # Report the learning rate used in the last optimization...
    lr_used = optimizer.param_groups[0]['lr']
    logger.info(f"MSG (device:{device}) - epoch {epoch}, lr used = {lr_used}")

    # Update learning rate in the scheduler...
    if scheduler is not None: scheduler.step(validate_loss_mean)


    # ___/ SAVE CHECKPOINT??? \___
    if validate_loss_mean < loss_min:
        loss_min = validate_loss_mean

        if (epoch % chkpt_saving_period == 0) or (epoch > epoch_unstable_end):
            fl_chkpt   = f"{timestamp}.epoch_{epoch}.chkpt"
            path_chkpt = os.path.join(drc_chkpt, fl_chkpt)
            save_checkpoint(model, optimizer, scheduler, epoch, loss_min, path_chkpt)
            logger.info(f"MSG (device:{device}) - save {path_chkpt}")


    # Shuffle the dataset...
    dataset_train.update_random_dataset()
    dataset_validate.update_random_dataset()

fh.close()
