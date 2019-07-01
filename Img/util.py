import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image


class MIMICDataLoader(Dataset):
    def __init__(self, image_list_file, transform):
        self.image_names = image_list_file
        self.transform = transform

    def __getitem__(self, index):     
        try:
            image_name = self.image_names[index]
            image = Image.open(image_name).convert('RGB')
            image = self.transform(image)
            return image
        except: print(image_name)

    def __len__(self):
        return len(self.image_names)



class MIMICTrainer():  
    def test(model, dataLoaderTest, checkpoint):   
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

        pred = torch.FloatTensor()
        
        model.eval()
        
        with torch.no_grad():
            for i, (input) in enumerate(dataLoaderTest):
                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w)
                out = model(varInput)
                pred = torch.cat((pred, out), 0)
        
        return pred
        
        
class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        self.num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(self.num_ftrs, 1)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        out = self.densenet121(x)
        out = self.dropout(out)
        return out
    
def Transform():
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = []
    transform.append(transforms.Resize((224, 224)))
    transform.append(transforms.ToTensor())
    transform.append(normalize)  
    
    return transforms.Compose(transform)


class Grad_CAM():
    def __init__ (self, pathModel):
        model = DenseNet121()
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(pathModel, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        self.model = model
        self.model.eval()
        # initialize the weights
        self.weights = list(self.model.module.densenet121.features.parameters())[-2]
        # downsize image
        self.transformSequence = Transform()
    
    def generate (self, pathImageFile):
        
        # load image, transform, convert 
        with torch.no_grad():
            imageData = Image.open(pathImageFile).convert('RGB')
            imageData = self.transformSequence(imageData)
            imageData = imageData.unsqueeze_(0)
            l = self.model(imageData)
            output = self.model.module.densenet121.features(imageData)
            # generate grad-cam image
            cam = None
            for i in range (0, len(self.weights)):
                map = output[0,i,:,:]
                if i == 0: cam = self.weights[i] * map
                else: cam += self.weights[i] * map
                npcam = cam.cpu().data.numpy()

        # blend original and grad-cam  
        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (224, 224))
        
        cam = npcam / np.max(npcam)
        cam = cv2.resize(cam, (224, 224))
        cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        
        img = cv2.addWeighted(imgOriginal,1,cam,0.35,0)            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

