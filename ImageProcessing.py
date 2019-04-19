import numpy as np
import torch
from PIL import Image
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Read csv
data = pd.read_csv('SKIN_CANCER_metadata.csv')

##Read an image to get resolution
directory = 'D:/git/HAM10000_images/' + data['name'][0] + '.jpg'
im = Image.open(directory, 'r')
pixels = im.size[0] * im.size[1]

#Create array to store pixel values of a batch (N x d)
images = torch.zeros(int((len(data['name']) - 15)/100), pixels)

#For each batch
for j in range(100):
    #For each image in a batch
    for i in range(int((len(data['name']) - 15)/100)):
        directory = 'D:/git/HAM10000_images/' + data['name'][100*j + i] + '.jpg'
        im = Image.open(directory, 'r')
        pix_val = list(im.getdata())
        #Flatten the image and convert to grayscale values from RGB
        pix_val_flat = [sum(sets)/3 for sets in pix_val]
        images[i, :] = torch.tensor(pix_val_flat)
    print("Saving file " + str(j))
    #Save the batch file
    np.savetxt('D:/git/Machine-Learning-Algorithms/batches/batch' + str(j) + '.csv', images.numpy(), delimiter = ",")