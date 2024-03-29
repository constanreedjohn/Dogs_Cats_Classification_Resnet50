# Dogs_Cats_Classification_Resnet50
 A project for dogs and cats classification using Resnet50 with Pytorch transfer learning.

# Instruction:
## Delete the text file in the following path:
 * ./dataset/Put_Dataset_Here.txt
 * ./saved_model/Put_Pretrained_Here.txt

## Requirements:
* pip install -r requirements.txt

## Transfer learning:
### Put dataset in the following structure:
* | ./dataset
* |== ./train
* |==== ./class_0
* |====== ./img_1.jpg
* |====== ./img_2.jpg
* |==== ./class_1
* |====== ./img_1.jpg
* |====== ./img_2.jpg
* |== ./val
* |==== ./class_0
* |====== ./img_1.jpg
* |====== ./img_2.jpg
* |==== ./class_1
* |====== ./img_1.jpg
* |====== ./img_2.jpg
* |== ./test
* |==== ./img_1.jpg
* |==== ./img_2.jpg

## TRAIN:
### Run train.py
* python train.py --help --> To see arguments
* python train.py --path ./dataset --batch_size 64 --num_epochs 10 --save_period 2 --save_path ./saved_model --name YourProjectName

## EVALUATION:
### Run test.py
* python test.py --help --> To see arguments
* python test.py --model_path ./pretrained.pt --path ./dataset --batch_size 64

## INFERENCE:
### Run infer.py
* python infer.py --help --> To see arguments
* python infer.py --model_path ./pretrained.pt --img_path ./infer_images --output_path ./output

# NOTE:
* Model must have the same architect from train to test to infer phase.
