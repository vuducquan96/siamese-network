import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import time
import numpy as np
import math

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
        return F.log_softmax(x, dim=1)

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
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),2))


        return loss_contrastive

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
def f_softmax(z,scale):
    #z_exp = [math.exp(i) for i in z]
    z_exp = np.array([]);
    sum_exp=0;
    for i in range(len(z)):
        z_exp = np.append(z_exp,math.exp(z[i]));
        sum_exp+=math.exp(z[i]);
    softmax = [i / sum_exp for i in z_exp]
    softmax= np.array([]);
    for i in range(len(z)):
        softmax= np.append(softmax,scale*(z_exp[i]/sum_exp));
    return softmax;

def main():
    """
    Initialize...
    """
    print(">>>>>>>>");
    print("Data loader:");
    #aaa=np.array([1,2]);
    #logsoftmax(aaa);
    total_size=90;
    model = SiameseNetwork().cuda();
    model.load_state_dict(torch.load("lala.ckpt"))
    data_test = torch.zeros(total_size, 1, 1, 100, 100).float();
    temp = torch.zeros(1, 1, 100, 100).float().cuda();
    link = "oneshot/s";
    tt=0;
    now=time.time();
    criterion = ContrastiveLoss()
    for count in range(1,11):
        for id in range(1,10):
            hey=link+str(count)+"/"+str(id)+".pgm";
            #print(hey);
            image = cv2.imread(hey, -1);
            image = cv2.resize(image, (100, 100));
            #cv2.imshow("image", image);
            #cv2.waitKey(0);
            #cv2.destroyAllWindows();
            image = torch.from_numpy(image).float();
            data_test[tt,0,0,:,:]=image;
            tt+=1;
    print("Time load data:{:.2f}".format(time.time()-now));
    lala = "testoneshot/s";
    tt=0;
    for count in range(4,5):
        link = lala + str(count) + "/" + "10.pgm";
        #print(link);
        image = cv2.imread(link, -1);
        image = cv2.resize(image, (100, 100));
        #cv2.imshow("image", image);
        #cv2.waitKey(0);
        #cv2.destroyAllWindows();
        temp[0,0,:,:]=torch.from_numpy(image).float().cuda();
        ans= np.array([]);
        tt = 0;
        for aa in range(1, 11):
            avr_loss=0;
            min = 999999.0;
            for id in range(1, 10):
                output1, output2 = model(data_test[tt,:,:,:,:].cuda(),temp);
                loss = criterion(output1, output2, 0);
                #loss=F.pairwise_distance(output1, output2)
                if (min>loss.item()):
                    min=loss.item();
                print(loss.item());
                avr_loss+=loss.item();
                tt += 1;
            #ans =np.append(ans,avr_loss/9);
            ans =np.append(ans,min);
        #print(ans);
        ans=f_softmax(ans,100);
        print(ans);
        print(ans.argmin()+1);
        print(">>>> aaa <<<<<");
    print(">>> Done <<<");

if __name__ == "__main__":
    main();