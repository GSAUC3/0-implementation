# Classification of Retinal Fundus Images for Diabetic Retinopathy using Deep Learning

<p align="center">
Department of Applied Optics & Photonics
</p>
<p align="center">
University College of Science, Technology & Agriculture
</p>
<p align="center">
University of Calcutta
</p>

Thesis submitted for the partial fulfilment of the requirement of
Department of Applied Optics and Photonics, 
University of Calcutta
For
The Degree of Bachelor of Technology
In
Optics and Optoelectronics Engineering


### Problem Statement
3 class classification of retinal fundus images. 

### Dataset: `Messidor`

Total number of images in the dataset: 1200

Diabetic Retinopathy Grade: 
|DR Grade|Description| Number of images 
|--------|:----:|:-----------------:|
|R0     | 𝜇𝐴 = 0 & H = 0 |546 |
|R1     | (0 < 𝜇𝐴 ≤ 5) & H = 0|153 |
|R2     |(5 < 𝜇𝐴 ≤ 15) OR (0 < 𝐻 < 5) & NV = 0 |247 |
|R3     | (𝜇𝐴 ≥ 15)  OR  (H ≥ 5)  OR  (NV = 1)|254 |
|Total   |  | 1200|

| Notations  | Description   |
|:---:|:----:|
|𝜇𝐴 | Number of microaneurysms|
|H | Number of Hemorrhages|
|NV =0 | NO neovascularization|
|NV =1 | Neovascularization|

The size of the fundus image is *1440 × 960*, *2240 × 1488*, or *2304 × 1536*.

|Dimensions (H,W,C)|No. of Images|
|:-:|:-:|
|(1488, 2240, 3)| 400|
|(960, 1440, 3)| 588|
|(1536, 2304, 3) |212|
|Total|1200|

`(H,W,C)` (Height, Width, No. of Channels)

`NOTE`: 13 duplicate images were found in the dataset which have been removed to avoid uncessary computation. 

`Meddior dataset consists of 4 Diabetic Retinopathy(DR) grade [DR 0, DR 1,DR 2, DR 3]. But in this project DR 0 and DR1 have been merged together to make single DR grade, (DR 0)`

- All the images have been cropped and reshaped to `512 x 512 x 3`

Following augmentation techniques were used to augment the training samples:
- Top-bottom flip
- Right-Left flip
- Rotation 150 deg
- Rotation 250d deg

So the new dataset becomes: 
|DR Grade|Description| Number of images 
|--------|:----:|:-----------------:|
|R0     |  (0 ≤ 𝜇𝐴 ≤ 5) & H = 0 |695|
|R1     |(5 < 𝜇𝐴 ≤ 15) OR (0 < 𝐻 < 5) & NV = 0  |240|
|R2     |(𝜇𝐴 ≥ 15)  OR  (H ≥ 5)  OR  (NV = 1)|252 |
||Total  |   1187|

### Training, Validation and Test set

|Sets|Number of images|
|:---:|:---:|
|Training set|831|
|Validation set|120|
|Test set|236|
|Total|1187|


### CNN architecture 

Efficient Net B0 with parameter reduction

Normally EfficientNet B0 has 5,288,548 trainable parameters. We can reduce it by discarding a few layers. Given below the implementation of the same.

```python
from torch import nn
from torchvision import models

model =tv.models.efficientnet_b0(pretrained=False)
model.features = model.features[:4]
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(in_features=40,out_features=3,bias=True)
)
print(model)
```
output:
```
EfficientNet(
  (features): Sequential(
    (0): ConvNormActivation(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): SiLU(inplace=True)
    )
    (1): Sequential(
      (0): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (2): ConvNormActivation(
            (0): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.0, mode=row)
      )
    )
    (2): Sequential(
      (0): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
            (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.0125, mode=row)
      )
      (1): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
            (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.025, mode=row)
      )
    )
    (3): Sequential(
      (0): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
            (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.037500000000000006, mode=row)
      )
      (1): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
            (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.05, mode=row)
      )
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=1)
  (classifier): Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=40, out_features=3, bias=True)
  )
)
```

```python
from torchsummary import summary
print(summary(model.cuda(),(3,512,512)))
```
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 256, 256]             864
       BatchNorm2d-2         [-1, 32, 256, 256]              64
              SiLU-3         [-1, 32, 256, 256]               0
            Conv2d-4         [-1, 32, 256, 256]             288
       BatchNorm2d-5         [-1, 32, 256, 256]              64
              SiLU-6         [-1, 32, 256, 256]               0
 AdaptiveAvgPool2d-7             [-1, 32, 1, 1]               0
            Conv2d-8              [-1, 8, 1, 1]             264
              SiLU-9              [-1, 8, 1, 1]               0
           Conv2d-10             [-1, 32, 1, 1]             288
          Sigmoid-11             [-1, 32, 1, 1]               0
SqueezeExcitation-12         [-1, 32, 256, 256]               0
           Conv2d-13         [-1, 16, 256, 256]             512
      BatchNorm2d-14         [-1, 16, 256, 256]              32
           MBConv-15         [-1, 16, 256, 256]               0
           Conv2d-16         [-1, 96, 256, 256]           1,536
      BatchNorm2d-17         [-1, 96, 256, 256]             192
             SiLU-18         [-1, 96, 256, 256]               0
           Conv2d-19         [-1, 96, 128, 128]             864
      BatchNorm2d-20         [-1, 96, 128, 128]             192
             SiLU-21         [-1, 96, 128, 128]               0
AdaptiveAvgPool2d-22             [-1, 96, 1, 1]               0
           Conv2d-23              [-1, 4, 1, 1]             388
             SiLU-24              [-1, 4, 1, 1]               0
           Conv2d-25             [-1, 96, 1, 1]             480
          Sigmoid-26             [-1, 96, 1, 1]               0
SqueezeExcitation-27         [-1, 96, 128, 128]               0
           Conv2d-28         [-1, 24, 128, 128]           2,304
      BatchNorm2d-29         [-1, 24, 128, 128]              48
           MBConv-30         [-1, 24, 128, 128]               0
           Conv2d-31        [-1, 144, 128, 128]           3,456
      BatchNorm2d-32        [-1, 144, 128, 128]             288
             SiLU-33        [-1, 144, 128, 128]               0
           Conv2d-34        [-1, 144, 128, 128]           1,296
      BatchNorm2d-35        [-1, 144, 128, 128]             288
             SiLU-36        [-1, 144, 128, 128]               0
AdaptiveAvgPool2d-37            [-1, 144, 1, 1]               0
           Conv2d-38              [-1, 6, 1, 1]             870
             SiLU-39              [-1, 6, 1, 1]               0
           Conv2d-40            [-1, 144, 1, 1]           1,008
          Sigmoid-41            [-1, 144, 1, 1]               0
SqueezeExcitation-42        [-1, 144, 128, 128]               0
           Conv2d-43         [-1, 24, 128, 128]           3,456
      BatchNorm2d-44         [-1, 24, 128, 128]              48
  StochasticDepth-45         [-1, 24, 128, 128]               0
           MBConv-46         [-1, 24, 128, 128]               0
           Conv2d-47        [-1, 144, 128, 128]           3,456
      BatchNorm2d-48        [-1, 144, 128, 128]             288
             SiLU-49        [-1, 144, 128, 128]               0
           Conv2d-50          [-1, 144, 64, 64]           3,600
      BatchNorm2d-51          [-1, 144, 64, 64]             288
             SiLU-52          [-1, 144, 64, 64]               0
AdaptiveAvgPool2d-53            [-1, 144, 1, 1]               0
           Conv2d-54              [-1, 6, 1, 1]             870
             SiLU-55              [-1, 6, 1, 1]               0
           Conv2d-56            [-1, 144, 1, 1]           1,008
          Sigmoid-57            [-1, 144, 1, 1]               0
SqueezeExcitation-58          [-1, 144, 64, 64]               0
           Conv2d-59           [-1, 40, 64, 64]           5,760
      BatchNorm2d-60           [-1, 40, 64, 64]              80
           MBConv-61           [-1, 40, 64, 64]               0
           Conv2d-62          [-1, 240, 64, 64]           9,600
      BatchNorm2d-63          [-1, 240, 64, 64]             480
             SiLU-64          [-1, 240, 64, 64]               0
           Conv2d-65          [-1, 240, 64, 64]           6,000
      BatchNorm2d-66          [-1, 240, 64, 64]             480
             SiLU-67          [-1, 240, 64, 64]               0
AdaptiveAvgPool2d-68            [-1, 240, 1, 1]               0
           Conv2d-69             [-1, 10, 1, 1]           2,410
             SiLU-70             [-1, 10, 1, 1]               0
           Conv2d-71            [-1, 240, 1, 1]           2,640
          Sigmoid-72            [-1, 240, 1, 1]               0
SqueezeExcitation-73          [-1, 240, 64, 64]               0
           Conv2d-74           [-1, 40, 64, 64]           9,600
      BatchNorm2d-75           [-1, 40, 64, 64]              80
  StochasticDepth-76           [-1, 40, 64, 64]               0
           MBConv-77           [-1, 40, 64, 64]               0
AdaptiveAvgPool2d-78             [-1, 40, 1, 1]               0
          Dropout-79                   [-1, 40]               0
           Linear-80                    [-1, 3]             123
================================================================
Total params: 65,853
Trainable params: 65,853
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.00
Forward/backward pass size (MB): 608.27
Params size (MB): 0.25
Estimated Total Size (MB): 611.52
----------------------------------------------------------------

```

### Other Parameters

- Optimizer: `Adam`
- Learning Rate: `5e-5`
- Batch size: `16`
- Loss function: `Cross Entropy Loss`
- Epochs: `80`
- Mean of the dataset (for normalization): `[0.4640, 0.2202, 0.0735]`
- Standard deviation of the dataset (for normalization): `[0.3127, 0.1529, 0.0554]`

### Results

<span style="color:Lime">Training Accuracy and Loss</span>

<span style="color:blue">Validation Accuracy and Loss.</span>

|Accuracy|Loss|
|:---:|:---:|
|<img src="https://github.com/GSAUC3/FYP/blob/master/images/acc.png">|<img src="https://github.com/GSAUC3/FYP/blob/master/images/loss.png">|
|Test accuracy|Test Loss|
|`88.98305084745762`| `0.336738`|


### Confusion Matrix

 <img src="https://github.com/GSAUC3/FYP/blob/master/images/cm.png">


#### 5 fold cross validation was performed on this dataset, the results are given below

|FOLDS|	ACCURACY ON TEST SET|	TEST LOSS|
|:---:|:---:|:---:|
|1	|88.98%|	0.336|
|2	|86.86%|	0.420|
|3	|81.77%|	0.599|
|4	|82.62%|	0.577|
|5	|84.32%|	0.5767|
|MEAN	|`84.91%`|	`0.5077`|
|STANDARD DEVIATION	|2.9925|	0.1069|


## Full Code

```python
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,models
from torchvision.datasets import ImageFolder
from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Use GPU if it's available or else use CPU.
print(device) #Prints the device we're using.


mean,std = [0.4640, 0.2202, 0.0735], [0.3127, 0.1529, 0.0554]
train = ImageFolder('generated/DATASET/train/',transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean,std)]))
val = ImageFolder('generated/DATASET/val/',transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean,std)]))
test = ImageFolder('generated/DATASET/test/',transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean,std)]))

model = models.efficientnet_b0(True)
model.features = model.features[:4]
model.classifier = nn.Sequential(nn.Dropout(0.5,inplace=True),
                            nn.Linear(40,4,bias=True))
batch=16
train_dl = DataLoader(train,batch,shuffle=True,num_workers=4)
val_ds = DataLoader(val,batch,num_workers=4,shuffle = False)
test_dl = DataLoader(test,batch,num_workers=4,shuffle = False)

def train(dataloader,model,loss_fn,optimizer):
    model.train()
    total= 0 
    correct = 0
    running_loss = 0

    for (x,y) in tqdm(dataloader):
        output = model(x.to(device))
        loss = loss_fn(output,y.to(device))

        running_loss += loss.item()
        total += y.size(0)
        predictions = output.argmax(dim=1).cpu().detach()
        # index of the highest score for all the samples in the batch
        correct += (predictions==y.cpu().detach()).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc=100*(correct/total)
    avg_loss = running_loss/len(dataloader) # average loss for a single batch
    print(f'\nTraining Loss = {avg_loss:.6f}',end='\t')
    print(f'Accuracy on Training set = {100*(correct/total):.6f}% [{correct}/{total}]') #Prints the Accuracy.
    
    return avg_loss,acc

def validate(dataloader,model,loss_fn):

#     model in evaluation mode
    model.eval() 
    
    total = 0
    correct = 0
    running_loss = 0
    
    with torch.no_grad(): # gradients calculation not required
        
        for x,y in tqdm(dataloader):
            
            output        = model(x.to(device)) #model's output.

            loss          = loss_fn(output,y.to(device)).item() #loss calculation.
            running_loss += loss
            
            total        += y.size(0)
            predictions   = output.argmax(dim=1).cpu().detach()
            correct      += (predictions == y.cpu().detach()).sum().item()
            
    avg_loss = running_loss/len(dataloader) #Average loss per batch.      
    val_acc = 100*(correct/total)
    print(f'\nValidation Loss = {avg_loss:.6f}',end='\t')
    print(f'Accuracy on Validation set = {100*(correct/total):.6f}% [{correct}/{total}]') #Prints the Accuracy.
    
    return avg_loss,val_acc

def optimize(train_dataloader,valid_dataloader,model,
loss_fn,optimizer,nb_epochs):
   
    train_losses = []
    valid_losses = []
    val= []
    acc=[]

    for epoch in range(nb_epochs):
        print(f'\nEpoch {epoch+1}/{nb_epochs}')
        print('-------------------------------')
        train_loss,a = train(train_dataloader,model,loss_fn,optimizer) 
        train_losses.append(train_loss)
        valid_loss,val_acc = validate(valid_dataloader,model,loss_fn)
        valid_losses.append(valid_loss)
        val.append(val_acc)
        acc.append(a)
    
    print('\nTraining has completed!')
    
    return train_losses,valid_losses,val,acc

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=5e-5) 
nb_epochs = 80
#Call the optimize function.
train_losses, valid_losses,v,acc = optimize(train_dl,val_ds,model,loss_fn,optimizer,nb_epochs)

epochs = range(nb_epochs)
plt.plot(epochs, train_losses, 'g', label='Training loss')
plt.plot(epochs, valid_losses, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, v, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

def Final_test(dataloader,model,loss_fn):

#     model in evaluation mode
    model.eval() 
    
    total = 0
    correct = 0
    running_loss = 0
    
    with torch.no_grad(): # gradients calculation not required
        
        for x,y in dataloader:
            
            output        = model(x.to(device)) #model's output.
            loss          = loss_fn(output,y.to(device)).item() #loss calculation.
            running_loss += loss
            
            total        += y.size(0)
            predictions   = output.argmax(dim=1).cpu().detach()
            correct      += (predictions == y.cpu().detach()).sum().item()
            
    avg_loss = running_loss/len(dataloader) #Average loss per batch.      
    val_acc = 100*(correct/total)
    print(f'test Loss = {avg_loss:.6f}',end='\t')
    print(f'Accuracy on test set = {100*(correct/total):.6f}% [{correct}/{total}]') #Prints the Accuracy.

Final_test(test_dl,model,loss_fn)
nb_classes = 3

# Initialize the prediction and label lists(tensors)
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

with torch.no_grad():
    for i, (inputs, classes) in enumerate(test_dl):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Append batch prediction results
        predlist=torch.cat([predlist,preds.view(-1).cpu()])
        lbllist=torch.cat([lbllist,classes.view(-1).cpu()])

# Confusion matrix
cm=confusion_matrix(lbllist.numpy(), predlist.numpy())
print(cm)

# Per-class accuracy
class_accuracy=100*cm.diagonal()/cm.sum(1)
print(class_accuracy)

print(classification_report(lbllist,predlist))
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return (diagonal_sum / sum_of_all_elements )*100
accuracy(cm)


confusion_matrix = np.zeros((nb_classes, nb_classes))
with torch.no_grad():
    for i, (inputs, classes) in enumerate(test_dl):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

plt.figure(figsize=(15,10))

# class_names = list(label2class.values())
class_names = ['DR 0','DR 1','DR 2']
df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
plt.ylabel('True label')
plt.xlabel('Predicted label')


```