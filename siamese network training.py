import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import torch.optim as optim
import random

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 30, kernel_size=3, stride=1)
        self.batch1 = nn.BatchNorm2d(30);
        self.conv2 = nn.Conv2d(30, 20, kernel_size=3, stride=1)
        self.batch2 = nn.BatchNorm2d(20);
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, stride=1)
        self.batch3 = nn.BatchNorm2d(20);
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(10580, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(2000, 2000)
        self.fc4 = nn.Linear(2000, 600)


    def forward_once(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = self.batch1(x);
        x = F.relu(F.avg_pool2d(self.conv2(x),2))
        x = self.batch2(x);
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = self.batch3(x);

        x = x.view(-1, 10580)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.dropout(x, training=0.0)
        x = self.fc4(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def main():
    """
    Initialize...
    """
    print(">>>>>>>>");
    epoch=1;
    learning_rate=0.00001;
    num_per_obj=9;
    num_obj=10;
    draw=200;
    loss=[];
    model = SiameseNetwork().cuda();
    total_size=num_per_obj*num_obj;
    print("Start.....");
    data_train = torch.zeros(total_size, 1, 1, 100, 100).float();
    link = "oneshot/s";
    tt=0;
    now=time.time();
    for count in range(1,num_obj+1):
        for id in range(1,num_per_obj+1):
            hey=link+str(count)+"/"+str(id)+".pgm";
            #print(link);
            image = cv2.imread(hey, -1);
            image = cv2.resize(image, (100, 100));
            image = torch.from_numpy(image).float();
            data_train[tt,0,0,:,:]=image;
            tt+=1;
    print("Time load data:{:.2f}".format(time.time()-now));
    #output1 ,output2= model(,);
    #load image in hear
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate);
    count = 0;
    current_loss = 0;
    now = time.time();
    arr = np.arange(total_size, dtype=np.uint32);
    #np.random.shuffle(arr);
    #print(arr);
    model.load_state_dict(torch.load("lala.ckpt"))
    for i in range(epoch):
        print("Shuffle Shuffle Shuffle");
        np.random.shuffle(arr);
        for bx in range(total_size-1):
            for by in range(bx+1,total_size):
                #print("{} {}".format(bx,by));
                ix=arr[bx];
                iy=arr[by];
                xx = int((ix+1) / num_per_obj - (float(int((ix+1) / num_per_obj)) == (ix+1) / num_per_obj));
                yy = int((iy+1) / num_per_obj - (float(int((iy+1) / num_per_obj)) == (iy+1) / num_per_obj));
                optimizer.zero_grad();
                output1, output2 = model(data_train[ix,:,:,:,:].cuda(),data_train[iy,:,:,:,:].cuda());
                label=(xx!=yy);
                #print(label);
                loss = criterion(output1,output2,label);
                loss.backward();
                optimizer.step();
                count+=1;
                #print("{} {} {}".format(ix,iy,label));
                current_loss+=loss.item();
                if (count%draw==0):
                    print("Training epoch:{} Time: {:.2f} Loss:{:.10f}".format(i, time.time() - now, current_loss/draw));
                    current_loss=0;
                    now=time.time();
    print(">>> Done <<<");
    torch.save(model.state_dict(), 'lala.ckpt');

if __name__ == "__main__":
    main();
