from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import json
import os
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F


class CNN(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


def Q1():
    print("Opening the picture Q1.jpg and transfer to grey")
    pil_im = Image.open('Q1.jpg').convert('L') #convert the grey image
    data1 = np.array(pil_im)
    #print(data1.shape) #(300, 276)
    data2 = data1.reshape((1, -1, pil_im.height, pil_im.width))
    img = torch.tensor(data=data2, dtype=torch.float, device=None, requires_grad=False)
    #print(img)
    horizontal = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    vertical = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    filter_horizontal = torch.tensor([[horizontal]], dtype=torch.float)
    filter_vertical = torch.tensor([[vertical]], dtype=torch.float)
    #torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
    dim = (1, 1, 1, 1)
    padding = nn.functional.pad(img, pad = dim, mode="replicate")
    res_horizontal = nn.functional.conv2d(padding, filter_horizontal, padding=0) #cannot reshape array of size 81652 into shape (300,276)
    res_vertical = nn.functional.conv2d(padding, filter_vertical, padding=0)
    res_relu_horizontal = torch.relu(res_horizontal)
    res_relu_vertical = torch.relu(res_vertical)
    res_pool_horizontal = torch.max_pool2d(res_relu_horizontal, kernel_size=2, stride=2)
    res_pool_vertical = torch.max_pool2d(res_relu_vertical, kernel_size=2, stride=2)


    padim_horizontal = Image.fromarray(res_horizontal.numpy().reshape(res_horizontal.shape[2], res_vertical.shape[3]))
    padim_vertical = Image.fromarray(res_vertical.numpy().reshape(res_vertical.shape[2], res_vertical.shape[3]))
    padim_horizontal_relu = Image.fromarray(res_relu_horizontal.numpy().reshape(res_relu_horizontal.shape[2], res_relu_horizontal.shape[3]))
    padim_vertical_relu = Image.fromarray(res_relu_vertical.numpy().reshape(res_relu_vertical.shape[2], res_relu_vertical.shape[3]))
    padim_horizontal_pool = Image.fromarray(res_pool_horizontal.numpy().reshape(res_pool_horizontal.shape[2], res_pool_horizontal.shape[3]))
    padim_vertical_pool = Image.fromarray(res_pool_vertical.numpy().reshape(res_pool_vertical.shape[2], res_pool_vertical.shape[3]))

    padim_horizontal = padim_horizontal.convert("RGB")
    padim_vertical = padim_vertical.convert("RGB")
    padim_horizontal_relu = padim_horizontal.convert("RGB")
    padim_vertical_relu = padim_horizontal.convert("RGB")
    padim_horizontal_pool  = padim_horizontal.convert("RGB")
    padim_vertical_pool  = padim_horizontal.convert("RGB")

    padim_horizontal_relu.save("Q1_horizontal_relu.jpg")
    padim_vertical_relu.save("Q1_vertical_relu.jpg")
    padim_horizontal_pool.save("Q1_horizontal_pool.jpg")
    padim_vertical_pool.save("Q1_vertical_pool.jpg")
    padim_horizontal.save("Q1_horizontal.jpg")
    padim_vertical.save("Q1_vertical.jpg")
    print("Pictures has been saved to local")


def Q(model, question):

    num_epochs = 10
    batch_size = 50
    learning_rate = 1e-3

    device = torch.device("cpu")
    #Loading the data set
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./fashionmnist_data/', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./fashionmnist_data/', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_epoch = []
    train_batch = []
    train_accuracy = []
    train_loss = []
    test_epoch = []
    test_batch = []
    test_accuracy = []
    test_loss_list = []

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
            return test_loss, float(correct) / float(len(test_loader.dataset))

    def train():
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(images), len(train_loader.dataset),
                        100. * i / len(train_loader), loss.item()))
            if (epoch*len(train_loader)+i) % 100 == 0:
                pred = outputs.data.max(1)[1]
                correct = pred.eq(labels.data.view_as(pred)).sum()
                train_epoch.append(epoch)
                train_batch.append(i * len(images))
                train_accuracy.append(float(correct) / batch_size)
                train_loss.append(float(loss.item()))
            if (epoch*len(train_loader)+i) % 1000 == 0:
                loss, accuracy = test()
                test_epoch.append(epoch)
                test_batch.append(i * len(images))
                test_accuracy.append(accuracy)
                test_loss_list.append(loss)

    for epoch in range(num_epochs):
        #train(model, device, train_loader, optimizer, epoch)
        #test(model, device, test_loader)
        train()

    torch.save(model.state_dict(), question + "mnist_fc.pt")

    def save_prediction(model, device, test_loader, epoches):
        model.eval()
        test_loss = 0
        correct = 0
        preds = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                test_loss += F.cross_entropy(outputs, target, reduction="sum").item()  # sum up batch loss
                pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                preds.extend(pred.tolist())
                correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

        test_epoch.append(epoches - 1)
        test_batch.append(len(train_loader.dataset))
        test_accuracy.append(float(correct) / float(len(test_loader.dataset)))
        test_loss_list.append(test_loss)
        with open(question +"saved_prediction.json", "w") as file:
            json.dump(preds, file)

    save_prediction(model, device, test_loader, num_epochs)

    train_x = [train_epoch[i] * len(train_loader.dataset) + train_batch[i] for i in range(len(train_loss))]
    test_x = [test_epoch[i] * len(train_loader.dataset) + test_batch[i] for i in range(len(test_loss_list))]

    plt.figure(figsize=(30, 15))
    plt.subplot(212)
    plt.title("Accuracy")
    plt.plot(train_x, train_accuracy, label="train")
    plt.plot(test_x, test_accuracy, label="test")
    plt.legend()

    plt.subplot(211)
    plt.title("Loss")
    plt.plot(train_x, train_loss, label="train")
    plt.plot(test_x, test_loss_list, label="test")
    plt.legend()


    plt.savefig(question +"history.png")


    idx, (data, target) = next(enumerate(test_loader))
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    pred = output.data.max(1)[1].tolist()
    target = target.tolist()
    plt.figure(figsize=(15, 15))
    for i in range(5):
        for j in range(5):
            plt.subplot2grid((5, 5), (i, j))
            plt.title("Pred: {}, GT: {}".format(pred[i*5+j], target[i*5+j]))
            plt.imshow(data[i*5+j].cpu().numpy().reshape((28, 28)), cmap="Greys")
    plt.savefig(question +"prediction.png")

if __name__ == "__main__":

    print("     Question 1     ")
    print("   ")
    Q1()
    print("     Question 2     ")
    print("   ")
    model2 = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        torch.nn.Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        CNN(),
        torch.nn.Linear(in_features=196, out_features=10, bias=True)
    )
    Q(model2, "Q2")

    print("     Question 3     ")
    print("   ")
    model3 = model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 6, kernel_size=5, stride=1),
        torch.nn.BatchNorm2d(6),
        torch.nn.Tanh(),
        torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        torch.nn.Conv2d(6, 16, kernel_size=5, stride=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.Tanh(),
        torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=0),
        torch.nn.Conv2d(16, 120, kernel_size=5, stride=1),
        torch.nn.BatchNorm2d(120),
        torch.nn.Tanh(),
        CNN(),
        torch.nn.Linear(in_features=3*3*120, out_features=120, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(in_features=120, out_features=84, bias=True),
        torch.nn.Linear(in_features=84, out_features=10, bias=True)
    )
    Q(model3, "Q3")
