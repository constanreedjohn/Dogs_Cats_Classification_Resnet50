import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def data_loader(path, batch_size):
    preprocess = {'train': transforms.Compose([
                                                transforms.Resize((256, 256)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]),
                  'valid': transforms.Compose([
                                                transforms.Resize((256, 256)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]),
                  'test': transforms.Compose([
                                                transforms.Resize((256, 256)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
                 }
    dataset = {x: ImageFolder(os.path.join(path, x), preprocess[x]) for x in ['train', 'valid', 'test']}
    loader = {"train": DataLoader(dataset["train"], batch_size=batch_size, shuffle=True),
              "valid": DataLoader(dataset["valid"], batch_size=batch_size, shuffle=True),
              "test": DataLoader(dataset["test"], batch_size=1, shuffle=False)}

    return loader