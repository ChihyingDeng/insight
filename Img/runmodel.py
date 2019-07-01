from Img.util import *


class RunModel():  
    def run(img_list):   
        # downsize image
        transformImg = Transform()

        # load image
        datasetTest = MIMICDataLoader(img_list, transformImg) 
        dataLoaderTest = DataLoader(dataset=datasetTest, num_workers=1, pin_memory=True, shuffle=False)

        # initialize the model
        model = DenseNet121()

        # load the model and get prediction
        pred = MIMICTrainer.test(model, dataLoaderTest, "Img/best.pth.tar")

        return pred
