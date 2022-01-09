import loader
import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import os
import argparse

def train(device, model, loader, criterion, num_epochs, save_path, save_period):
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # train
    for epoch in range(0, num_epochs):
        losses = []
        model.train()
        loop = tqdm(enumerate(loader), total=len(loader))
        for batch_idx, (data, target) in loop:
            data = data.to(device)
            target = target.to(device)
            scores = model(data)

            loss = criterion(scores, target)
            optimizer.zero_grad()
            losses.append(loss)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(scores, 1)

            loop.set_description(f"Epoch {epoch+1}/{num_epochs} process: {int((batch_idx / len(loader)) * 100)}")
            loop.set_postfix(loss=loss.data.item())

    # Save model
        if (save_period > -1) and (epoch+1 % save_period == 0):
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),

            }, f"{save_path}/Checkpoint_epoch_{str(epoch)}.pt")

    return model, criterion

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 2, bias=True)
    print("Model configured: ", model.fc)
    model.to(device)

    return model, device

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default= os.getcwd()+"/dataset", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=64, help="Data batch-size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--save_period", type=int, default=-1, help="Save every n_th epoch")
    parser.add_argument("--save_path", type=str, default= os.getcwd() + "/saved_model", help="Save model path")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_opt()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    data_loader = loader.data_loader(args.path, args.batch_size)
    model, device = main()
    CNN, crit = train(device, model, data_loader['train'], criterion, args.num_epochs, args.save_path, args.save_period)