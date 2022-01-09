import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import argparse
import os

def load_model(model_path):
    # Load checkpoint
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 2, bias=True)
    print("----> Loading checkpoint")
    checkpoint = torch.load(model_path, map_location=torch.device("cpu")) # Try to load last checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded")
    model.to(device)
    
    return model

def Inference(device, model, img_path, out_path):
    count = 0   

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Load image
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    for img_f in os.listdir(img_path):
        ext = img_f.split(".")[1]
        image_path = os.path.join(img_path, img_f)    
        img_array = Image.open(image_path).convert("RGB")
        # img_ls.append(img_array)
    
        temp = data_transforms(img_array).unsqueeze(dim=0)
        load = DataLoader(temp)

        for x in load:
            x = x.to(device)
            output = model(x)
            _, pred = torch.max(output, 1)

            # Show image
            image = cv2.imread(image_path)
            # print(f"Class: {pred}")
            if (pred[0] == 1): 
                cv2.imwrite(f"{out_path}/Dog{count}.{ext}", image)
            else: 
                cv2.imwrite(f"{out_path}/Cat{count}.{ext}", image)
            count += 1
            
            cv2.waitKey()
            cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default= os.getcwd()+"/saved_model/pretrained.pt", help="Trained model path")
    parser.add_argument("--img_path", type=str, default=os.getcwd()+"/infer_images", help="Image folder path")
    parser.add_argument("--out_path", type=str, default=os.getcwd(), help="Predicted Image folder path")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    args = parse_opt()
    model = load_model(args.model_path)
    Inference(device, model, args.img_path, args.out_path)