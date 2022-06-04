import torch
### YOUR CODE HERE

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

mnist_train = datasets.MNIST(root = "./datasets", train = True, transform= transforms.ToTensor(), download= False)
mnist_test = datasets.MNIST(root = "./datasets", train = False, transform= transforms.ToTensor(), download= False)
trainloader = torch.utils.data.DataLoader(mnist_train, batch_size = 100, shuffle = True)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size = 100, shuffle = True)

# Make sure to print out your accuracy on the test set at the end.

W = torch.randn(784, 500)/np.sqrt(784)
W.requires_grad_()
b = torch.zeros(500, requires_grad=True)
V = torch.randn(500, 10)/np.sqrt(500)
V.requires_grad_()
c = torch.zeros(10, requires_grad= True)

optimizer = torch.optim.SGD([V,c,W,b], lr=0.12)

for images, labels in tqdm(trainloader):

    optimizer.zero_grad()
    
    x = images.view(-1, 28*28)
    y = torch.matmul(x,W)+b

    y_relu_F = F.relu(y)

    z = torch.matmul(y_relu_F, V)+c
    
    cross_entropy = F.cross_entropy(z, labels)
    cross_entropy.backward()
    optimizer.step()

    correct = 0
    total = len(mnist_test)

with torch.no_grad():

    for images, labels in tqdm(testloader):
        x = images.view(-1, 28*28)
        y = torch.matmul(x,W)+b
            
        y_relu_F = F.relu(y)
            
        z = torch.matmul(y_relu_F, V)+c

        z = F.softmax(z, 1)

        predictions = torch.argmax(z, dim=1)

        correct += torch.sum((predictions == labels).float())
    
print('Test accuracy: {}'.format(correct/total))
