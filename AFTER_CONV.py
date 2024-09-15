import openpyxl
import torch
import torch.nn as nn
import torch.optim as optim
import os
import gc
from torchvision import transforms
from torch.utils.data import DataLoader
import tqdm
import pandas as pd
#
# from MODEL import COCOObjectDetectionDataset, MobileNetInspiredDetector, \
#     BoundingBoxProcessor,data_loader


def save_checkpoint(model, optimizer, epoch, batch_idx, loss, filename):
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)


def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        batch_idx = checkpoint['batch_idx']
        loss = checkpoint['loss']
        return model, optimizer, epoch, batch_idx, loss
    return model, optimizer, 0, 0, None
from tqdm import tqdm



def train_incrementallymod(model,loss_file, chkpt_file, bbox_processor, optimizer, imgs, tgt, num_epochs, checkpoint_interval,
                           max_batches_per_run, batch_indices):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_dict = {}
    start_epoch, start_batch, end_batch = batch_indices
    work = openpyxl.Workbook()
    sheet = work.active
    sheet.title = "UltraLoss_data"
    with open(loss_file, 'a') as f:
        # Write the header if needed
        f.write("Batch,Loss\n")
        for epoch in range(start_epoch, num_epochs):
            for batch_idx, images, targets in zip(range(start_batch, end_batch), imgs, tgt):
                try:
                    images = images.to(device)
                    # targets = targets.to(device)

                    optimizer.zero_grad()
                    predictions = model(images)
                    loss = bbox_processor.loss_forward(predictions=predictions, targets=targets)
                    loss.backward()
                    optimizer.step()
                    loss_dict[batch_idx] = loss.item()
                    f.write(f"{batch_idx},{loss.item()}\n")
                    f.flush()
                    # df = pd.DataFrame.from_dict(loss_dict, orient='index', columns=['Loss'])
                    # df = pd.DataFrame.from_dict(loss_dict, orient='index', columns=['Loss'])
                    # with open('loss.csv', 'a') as f:
                    #     df.to_csv(f)

                    if (batch_idx + 1) % checkpoint_interval == 0:
                        save_checkpoint(model, optimizer, epoch, batch_idx + 1, loss.item(), chkpt_file)
                        print(f"Checkpoint saved. Epoch {epoch}---Batch {batch_idx + 1}--- Loss: {loss.item()}")

                    if batch_idx + 1 >= max_batches_per_run+end_batch:
                        save_checkpoint(model, optimizer, epoch, batch_idx + 1, loss.item(), chkpt_file)
                        print(f"Max batches reached. Saving and exiting. Epoch---{epoch}--- Batch {batch_idx + 1}")

                        return False  # Exit to allow re-invocation (if necessary)

                except torch.cuda.OutOfMemoryError as e:
                    print(f"Out of memory error on GPU: {e}. Exiting training.")
                    return True  # Signal to stop the loop due to memory error

            # Reset batch for next epoch
            # start_batch = 0

    return True  # Indicate training is complete
from tqdm import tqdm

