import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from model import CordModel
from dataset import train_loader
import torch
import torch.nn.functional as F

model=CordModel()
optizmer=torch.optim.SGD(model.parameters(),lr=0.03)
loss_f=F.cross_entropy
batch_size=20

for epoch in range (10):
 model.train()
 for input_data,input_label in (train_loader):
  loss_data=0
  accuracy=0
  for i in range (batch_size):
   pred=model(input_data[i])
   loss=loss_f(pred,input_label[i])
   loss_data+=loss.item()
   accuracy+=(pred.argmax(1)==input_label[i].item())
   optizmer.zero_grad()
   loss.backward()
   optizmer.step()
  print(accuracy)
 print("*************epoch ends**********")
torch.save(model.state_dict(), 'my_new_model_.pth')