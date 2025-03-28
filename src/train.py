import time
import os
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torchvision.transforms.v2 as v2

# from src.model.model import UNet3D, ResidualUNet3D
from src.model.unet import UNet3D
from src.dataset import ACDC, ACDCProcessed, JustToTest
from src.loss import DiceLoss3D
from src.metrics import calculate_multiclass_dice

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def print_vram(): 
    print(f"Allocated: {torch.cuda.memory_allocated()/1e6} MB", flush = True)
    print(f"Reserved: {torch.cuda.memory_reserved()/1e6} MB", flush = True)

def setup(rank, world_size): 
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1749"
    
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup(): 
    dist.destroy_process_group()


def train(rank, world_size): 
    setup(rank, world_size)

    batch_size = 2
    EPOCHS = 50
    best_model_path = "model/unet3d_6.pth"
    model_path = "model/unet3d_6.pth"
    train_result_path = "train_result/train_result_6.csv"
    min_loss = 0
    

    train_dataset = JustToTest("just_to_test/training/", is_testset=False)
    valid_dataset = JustToTest("just_to_test/validation/", is_testset=True)

    model = UNet3D().to(rank)
    
    if (os.path.exists(best_model_path)): 
        model.load_state_dict(torch.load(best_model_path))
        
    model = DDP(model, device_ids=[rank])

    loss_fn = DiceLoss3D()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, foreach=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(valid_dataset, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers = 1,
        sampler=train_sampler
    )
    valid_dataloader = DataLoader(valid_dataset, 
        batch_size=batch_size, 
        num_workers = 1,
        sampler=val_sampler
    )

    train_loss = []
    val_loss = []

    # Start training
    for epoch in range(EPOCHS): 
        if (rank == 0): 
            start = time.time()
            print("-" * 50, flush = True)
            print(f"Epoch [{epoch+1}/{EPOCHS}]: ", flush = True)

        train_sampler.set_epoch(epoch)

        model.train()
        loss = train_one_epoch(rank, model, train_dataloader, loss_fn, optimizer)

        model.eval()
        vloss = eval_one_epoch(rank, model, valid_dataloader, loss_fn)
        scheduler.step(vloss)
        
        torch.distributed.barrier()
        
        if (rank == 0): 
            train_loss.append(loss)
            val_loss.append(vloss)

            print(f"Train loss: {loss}", flush = True)
            print(f"Validation Dice Coeff: {vloss}", flush = True)

            if (vloss > min_loss): 
                min_loss = vloss

                torch.save(model.module.state_dict(), model_path)
                torch.cuda.empty_cache()
                print(f"Save model to {model_path} in process {rank}")

            end = time.time()
            elapsed = (end - start)/60
            print(f"Time: {elapsed:3f} minutes")

    df = pd.DataFrame({
        "train_loss": train_loss, 
        "val_loss": val_loss
    })
    df.to_csv(f"{train_result_path}", index = False)


def train_one_epoch(rank, model, train_dataloader, loss_fn, optimizer): 
    running_loss = 0

    for i, data in enumerate(train_dataloader): 
        img, gt = data 

        img = img.to(rank)
        gt = gt.to(rank)

        optimizer.zero_grad(set_to_none = True)
        output = model(img)
        loss = loss_fn(output, gt)
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step() 

        running_loss += loss.item()
        if (i % 10 == 9 and rank == 0): 
            print(f"[{i+1}/{len(train_dataloader)}]", flush = True)
            print_vram()

        del loss, output, img, gt
        torch.cuda.empty_cache()

    return running_loss/len(train_dataloader)


def eval_one_epoch(rank, model, valid_dataloader, loss_fn): 
    running_vloss = 0
    running_vdice = 0
    batch_dice = []
    
    with torch.no_grad(): 
        for i, data in enumerate(valid_dataloader): 
            img, gt = data

            img = img.to(rank)           
            gt = gt.to(rank)

            output = model(img)
            loss = loss_fn(output, gt) 
            
            output = (output > 0.5).to(torch.long)
            
            for batch in range(output.size(0)): 
                dice_per_batch, _ = calculate_multiclass_dice(
                    output[batch, :,:,:,:], 
                    gt[batch, :,:,:,:]
                )
                batch_dice.append(dice_per_batch)
                
            dice = np.mean(batch_dice)
              
            running_vloss += loss.item()
            running_vdice += dice 
            
            if (i % 10 == 9 and rank == 0): 
                print(f"[{i+1}/{len(valid_dataloader)}]", flush = True)

    return running_vdice/len(valid_dataloader)


def main(): 
    world_size = torch.cuda.device_count()
    print("Device count:", world_size, flush = True)
    mp.spawn(
        train, 
        args=(world_size,), 
        nprocs=world_size, 
        join=True
    )

if __name__ == "__main__": 
    main()
