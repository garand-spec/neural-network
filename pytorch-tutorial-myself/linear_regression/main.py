import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

root = "../../pytorch-tutorial/tutorials/data"

input_size = 28 * 28 
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.01

model = nn.Linear(input_size, num_classes)

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root=root, 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=False)

test_dataset = torchvision.datasets.MNIST(root=root, 
                                          train=False, 
                                          transform=transforms.ToTensor(),
                                          download=False)

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if(i % 100 == 0):
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))