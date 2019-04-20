import numpy as np
import torch
from PIL import Image
import pandas as pd
import sys

#Read csv
data = pd.read_csv('SKIN_CANCER_metadata.csv')

##Read an image to get resolution
directory = '/Users/shivamodeka/Desktop/Machine-Learning-Algorithms/HAM10000/' + data['Name'][0] + '.jpg'
im = Image.open(directory, 'r')
pixels = im.size[0] * im.size[1]
batches = 100

if(int(len(data['Name']) % batches) > 0):
    print("Number of batches not a multiple of total data")
    sys.exit(1)

#Create array to store pixel values of a batch (N x d)
images = torch.zeros(int(len(data['Name'])/batches), pixels+4)

#For each batch
for j in range(batches):
    #For each image in a batch
    for i in range(int(len(data['Name'])/batches)):
#        i =0
#        j =0
        directory = '/Users/shivamodeka/Desktop/Machine-Learning-Algorithms/HAM10000/' + data['Name'][batches*j + i] + '.jpg'
        im = Image.open(directory, 'r')
        pix_val = list(im.getdata())
        #Flatten the image and convert to grayscale values from RGB
        pix_val_flat = [sum(sets)/3 for sets in pix_val]
        pix_val_flat.append(data['Age'][batches*j + i])
        pix_val_flat.append(data['Male'][batches*j + i])
        pix_val_flat.append(data['Female'][batches*j + i])
        pix_val_flat.append(data['Label'][batches*j + i])
        images[i, :] = torch.tensor(pix_val_flat)
    print("Saving file " + str(j))
    #Save the batch file
#    images = images/255
    np.savetxt('/Users/shivamodeka/Desktop/Machine-Learning-Algorithms/batches/batch' + str(j) + '.csv', images.numpy(), delimiter = ",")
    
