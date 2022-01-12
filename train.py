import loader
import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import os
import argparse
import time

def train(device, model, loader, criterion, optimizer, num_epochs, save_path, save_period):
    train = {'loss': [], 'acc': []}
    val = {'loss': [], 'acc': []}
    
    start = time.time()
    # train
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        for phase in ['train', 'valid']:
            loop = tqdm(enumerate(loader[phase]), total=len(loader[phase]))
            if phase == 'train':
                model.train()
            else: 
                model.eval()
            
            running_loss = 0.0
            running_acc = 0.0

            for batch_idx, (data, target) in loop:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    scores = model(data)
                    loss = criterion(scores, target)
                    _, pred = torch.max(scores, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    loop.set_description(f"{phase} process: {int(batch_idx / len(loader[phase]) * 100)}")
                    loop.set_postfix(loss=loss.data.item())
                elif phase == 'valid':
                    loop.set_description(f"{phase} process: {int(batch_idx / len(loader[phase]) * 100)}")
                    loop.set_postfix(loss=loss.data.item())
                
                running_loss += loss.item() * data.size(0)
                running_acc += torch.sum(pred == target.data)
            
            # Create statiscal result
            if phase == 'train':
                train['loss'].append(running_loss / len(loader[phase].dataset))
                train['acc'].append(running_acc.double() / len(loader[phase].dataset))
            elif phase == 'valid':
                val['loss'].append(running_loss / len(loader[phase].dataset))
                val['acc'].append(running_acc.double() / len(loader[phase].dataset))
            
            # Result per epoch
            epoch_loss = running_loss / len(loader[phase].dataset)
            epoch_acc = running_acc.double() / len(loader[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # Save model
        if (save_period != -1) and ((epoch+1) % save_period == 0):
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epochs': epoch,
            }, f"{save_path}/Checkpoint_epoch_{str(epoch)}.pt")

    end = time.time()
    elapse = end - start
    print(f"Training complete in {(elapse // 60):.0f}m {(elapse % 60):.2f}s")
    print()

    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model = torchvision.models.resnet50(pretrained=True)
    optimizer = Adam(model.parameters(), lr=0.001)
    model.fc = nn.Linear(2048, 2, bias=True)
    print("Model configured: ", model.fc)
    model.to(device)

    return device, model, optimizer

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
    device, model, optimizer = main()
    CNN = train(device, model, data_loader, criterion, optimizer, args.num_epochs, args.save_path, args.save_period)