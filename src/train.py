import gc
import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torchvision.transforms.v2 as v2

from src.model.model import UNet3D
from src.dataset import ACDC


def setup(rank, world_size): 
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1749"
    
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup(): 
    dist.destroy_process_group()


def train(rank, world_size): 
    setup(rank, world_size)

    batch_size = 1
    EPOCHS = 50
    
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # fold_results = []

    train_dataset = ACDC("database/training/")
    valid_dataset = ACDC("database/valid/")
    # global_min_loss = 9999

    # for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)): 
    #     if (rank == 0): 
    #         print(f"Fold {fold+1}/5")

    model = UNet3D(in_channels=1, out_channels=4, is_segmentation=False).to(rank)
    model = DDP(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, foreach=True)

    # train_subset = Subset(dataset, train_idx)
    # val_subset = Subset(dataset, val_idx)

    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    sampler=DistributedSampler(train_dataset))
    valid_dataloader = DataLoader(valid_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    sampler=DistributedSampler(valid_dataset))

    train_loss = []
    val_loss = []
    min_loss = 9999

    # Start training
    for epoch in range(EPOCHS): 
        if (rank == 0): 
            print("-" * 50)
            print(f"Epoch [{epoch+1}/{EPOCHS}]: ")

        train_dataloader.sampler.set_epoch(epoch)
        valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        loss = train_one_epoch(rank, model, train_dataloader, loss_fn, optimizer)
        avg_loss = average_loss(loss, world_size)

        model.eval()
        vloss = eval_one_epoch(rank, model, valid_dataloader, loss_fn)
        avg_vloss = average_loss(vloss, world_size)
        
        if (rank == 0): 
            train_loss.append(avg_loss)
            val_loss.append(avg_vloss)

            print(f"\tTrain loss: {avg_loss}")
            print(f"\tValidation loss: {avg_vloss}")

            if (avg_vloss < min_loss): 
                min_loss = avg_vloss

                model_path = "model/unet3d.pth"

                torch.save(model.module.state_dict(), model_path)
                print(f"Save model to {model_path} in process {rank}")

    # fold_results.append(min_loss)
    df = pd.DataFrame({
        "train_loss": train_loss, 
        "val_loss": val_loss
    })
    df.to_csv(f"train_result.csv")
    # del model, loss_fn, optimizer, train_dataloader, valid_dataloader
    # torch.cuda.empty_cache()
    # gc.collect()

    # if (rank == 0): 
    #     with open("train_result.txt", "w") as f: 
    #         f.write(f"Best model: {np.argmax(fold_results) + 1}")


def train_one_epoch(rank, model, train_dataloader, loss_fn, optimizer): 
    running_loss = 0

    for i, data in enumerate(train_dataloader): 
        img, gt = data 

        img = img.to(rank)
        gt = gt.squeeze(1).to(rank)
        
        optimizer.zero_grad(set_to_none = True)
        output = model(img)
        loss = loss_fn(output, gt)
        loss.backward()
        optimizer.step() 
        
        running_loss += loss.item()
        if (i % 10 == 9 and rank == 0): 
            print(f"[{i+1}/{len(train_dataloader)}]")

    return running_loss/len(train_dataloader)


def eval_one_epoch(rank, model, valid_dataloader, loss_fn): 
    running_vloss = 0
    
    with torch.no_grad(): 
        for i, data in enumerate(valid_dataloader): 
            img, gt = data

            img = img.to(rank)           
            gt = gt.squeeze(1).to(rank)

            output = model(img)
            loss = loss_fn(output, gt) 
            
            running_vloss += loss.item()
            if (i % 10 == 9 and rank == 0): 
                print(f"[{i+1}/{len(valid_dataloader)}]")

    return running_vloss/len(valid_dataloader)


def average_loss(loss, world_size): 
    '''
    Average loss in all processes

    Parameter: 
    --- 
    world_size (int): Number of processes
    '''
    loss_tensor = torch.tensor(loss, device = "cpu")
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return loss_tensor.item()/world_size


def main(): 
    world_size = torch.cuda.device_count()
    print("Device count:", world_size)
    mp.spawn(
        train, 
        args=(world_size,), 
        nprocs=world_size, 
        join=True
    )

if __name__ == "__main__": 
    main()
