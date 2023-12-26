import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import cv2 as cv;
# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt;
import numpy as np
DEVICE = "cpu"
NUM_CLASSES = 22
PATH_TO_MODEL = "https://drive.google.com/file/d/1OIA-pexNjin2YW8OV03Y8KRhqNhrfGRr/view?usp=sharing"
CLASSES = ['Cashew_anthracnose', 'Cashew_gumosis', 'Cashew_healthy', 'Cashew_leaf miner', 'Cashew_red rust', 'Cassava_bacterial blight', 'Cassava_brown spot', 'Cassava_green mite', 'Cassava_healthy', 'Cassava_mosaic', 'Maize_fall armyworm', 'Maize_grasshoper', 'Maize_healthy', 'Maize_leaf beetle', 'Maize_leaf blight', 'Maize_leaf spot', 'Maize_streak virus', 'Tomato_healthy', 'Tomato_leaf blight', 'Tomato_leaf curl', 'Tomato_septoria leaf spot', 'Tomato_verticulium wilt']

def loadImage(pathToImage):
    """"
    takes input the path of image and returns the tensor with the dimensions of batch
    """
    image = Image.open(pathToImage)

    transformToTorch = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    tensorImage = transformToTorch(image).to(DEVICE, dtype=torch.float32)

    batchTensor = tensorImage.unsqueeze(0)

    return batchTensor;

def loadModel(pathToModel):
    """
    takes input the path for .pth file of the (modified)resnet, loads weights and returns the model
    """
    
    resnet = models.resnet50()

    resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)

    resnet = resnet.to(DEVICE)

    checkpoint = torch.load(pathToModel,map_location=torch.device(DEVICE))                                                                              
    
    resnet.load_state_dict(checkpoint['model_state_dict'])

    return resnet;

def fgsm_attack(image, epsilon, gradient):
  gradient_sign = gradient.sign()
  noise = epsilon*gradient_sign;
  perturbed_image = image + noise
  return torch.clamp(perturbed_image,0,1),noise

def pgd_attack(model, image, epsilon, truth_tensor,displayNoise=False):
#   print("PGD attack ", truth_tensor)
  numIters = 10
  stepSize =2/255

  for _ in range(numIters):
    model.zero_grad()

    output = model(image)
    loss = nn.CrossEntropyLoss()(output, truth_tensor)
    loss.backward(retain_graph=True)
    gradient_sign = image.grad.sign()
    noise = epsilon * gradient_sign;
    perturbed_image = image + noise
    inputTensor = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
    inputTensor = perturbed_image.detach().requires_grad_()

  return inputTensor,noise

def displayTensorImage(tensor):
    tensor_to_pil = transforms.ToPILImage()(tensor)
    # display(tensor_to_pil)
    return np.array(tensor_to_pil)

def simulateAttack(inputPath,label,fgsm=False, epsilon=0.03,displayNoise=False):

    resnet = loadModel(PATH_TO_MODEL)
    inputTensor = loadImage(inputPath)

    criterion = nn.CrossEntropyLoss()
    DEVICE = 'cpu'
    resnet.to(DEVICE)

    truth_tensor = label;
    inputTensor.requires_grad = True;

    resnet.eval()

    output = resnet(inputTensor)

    confidenceTensor = torch.nn.Softmax(dim=1)(output).data
    maxProbIdx = torch.argmax(confidenceTensor,dim=1)
    loss = criterion(output, truth_tensor)

    # print(f"Output Label {classes[maxProbIdx]}\nConfidence Level {(confidenceTensor.tolist())[0][maxProbIdx]*100}%\nLoss {loss}")
    # print(f"Output Label {maxProbIdx}\nConfidence Level {confidenceTensor}%\nLoss {loss}")

    #until this point for both attacks the process will be the same
    resnet.zero_grad()
    loss.backward()
    gradients = inputTensor.grad

    if fgsm==True:
        inputTensor,noiseTensor = fgsm_attack(inputTensor,epsilon, gradients)
    else:
        inputTensor,noiseTensor = pgd_attack(resnet,inputTensor, epsilon, truth_tensor)

    attack_image=inputTensor.squeeze(0)
    noiseTensor = noiseTensor.squeeze(0)

    output_adversial = resnet(inputTensor)

    attackConfidenceTensor = torch.nn.Softmax(dim=1)(output_adversial).data

    maxAttackProbIdx = torch.argmax(attackConfidenceTensor,dim=1)

    # print(f"Output Label {classes[maxAttackProbIdx]}\nConfidence Level {(attackConfidenceTensor.tolist())[0][maxAttackProbIdx]*100}%")

    return (
        displayTensorImage(attack_image), # adversial image

        CLASSES[maxAttackProbIdx], #adversial class

        (attackConfidenceTensor.tolist())[0][maxAttackProbIdx]*100, # adversial confidence

        displayTensorImage(noiseTensor), # noise tensor

        CLASSES[maxProbIdx], # original class

        (confidenceTensor.tolist())[0][maxProbIdx]*100, #original confidence

        )


def inference(pathToImage):
    """"
    takes input the path for the image and returns the probabilities
    """
    tensor = loadImage(pathToImage);
    model = loadModel(pathToModel=PATH_TO_MODEL);
    
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        confidenceTesnor = torch.nn.Softmax(dim=1)(output).data
        maxProbIdx = torch.argmax(confidenceTesnor,dim=1)
        finalPrediction = CLASSES[maxProbIdx];
    
    # print(f"Disease {finalPrediction} Confidence Level {(confidenceTesnor.tolist())[0][maxProbIdx]*100}%\n");
    return (finalPrediction, format(confidenceTesnor.tolist()[0][maxProbIdx]*100,'.2f'),maxProbIdx);
    # imageFile = cv.imread(pathToImage,cv.IMREAD_UNCHANGED);
    # image_np = np.array(imageFile)
    # plt.imshow(image_np);
    # plt.show()



if __name__ == "__main__":
#    checkpoint = {
#         'model_state_dict': torch.load(PATH_TO_MODEL,map_location=torch.device('cpu'))['model_state_dict'],
#    }
#    torch.save(checkpoint, 'checkpoint_weights.pth')
    parser = argparse.ArgumentParser(description='Script for making infrences from model')
    parser.add_argument('-p', type=str, required=True, help='path to the image for which inference has to be made')
    args = parser.parse_args()

    print("Image path:", args.p)

    imagePath = args.p

    assert os.path.exists(imagePath), f"No file found at {imagePath}"

    inference(imagePath)
