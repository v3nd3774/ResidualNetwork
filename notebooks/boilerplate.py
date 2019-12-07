import torchvision
import os
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
from operator import mul as multiply
from IPython.display import clear_output


device = "cuda" if torch.cuda.is_available() else "cpu"
import pdb

def k_by_k_conv(f_0, f_1, k=3, stride=1, padding=1, bias=False):
    opts = {
        "in_channels": f_0,
        "out_channels": f_1,
        "kernel_size": k,
        "stride": 1,
        "padding":1,
        "bias": False
    }
    return nn.Conv2d(**opts)

def bn(dims): return nn.BatchNorm2d(dims)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = k_by_k_conv(self.in_channels, self.out_channels)
        self.bn1 = bn(self.out_channels)
        self.sigma1 = nn.ReLU(inplace=True)
        self.conv2 = k_by_k_conv(self.out_channels, self.out_channels)
        self.bn2 = bn(self.out_channels)
        self.sigma2 = nn.ReLU(inplace=True)
        self.block = nn.Sequential(self.conv1,
                                   self.bn1,
                                   self.sigma1,
                                   self.conv2,
                                   self.bn2,
                                   self.sigma2)
    def forward(self, t_1):
        return self.block(t_1)
    
class ResNetBasicBlock(BasicBlock):
    def __init__(self, in_channels, out_channels):
        super(ResNetBasicBlock, self).__init__(in_channels, out_channels)
        del self.block[-1]
        self.use_identity = self.in_channels == self.out_channels
        if not self.use_identity:
            downsample_opts = {"k":1,
                               "stride": 2}
            self.one_by_one_downsample = k_by_k_conv(in_channels, 
                                                     out_channels, 
                                                     **downsample_opts)
    def shortcut(self, t_1, t_5):
        if self.use_identity:
            return t_1 + t_5
        else:
            return self.one_by_one_downsample(t_1) + t_5
    def forward(self, t_1):
        t_5 = self.block(t_1)
        return self.sigma2(self.shortcut(t_1, t_5))
    
class Network(nn.Module):
    def __init__(self, block, n=7):
        super(Network, self).__init__()
        self.n = n
        self.sizes = [32, 16, 8]
        
        # 2n + 1 layers of 32x32 with 16 filters
        self.block_channel_sizes = [(3, 16)]
        for _ in range(2 * n):
            self.block_channel_sizes.append((16, 16))
            
        # 2n layers of 16x16 with 32 filters
        self.block_channel_sizes.append((16, 32))
        for _ in range(2 * n - 1):
            self.block_channel_sizes.append((32, 32))
            
        # 2n layers of 8x8 with 64 filters
        self.block_channel_sizes.append((32, 64))
        for _ in range(2 * n - 1):
            self.block_channel_sizes.append((64, 64))
            
        # glue all layers together
        self.layers = nn.Sequential(*[block(channel_in, channel_out)
                                     for channel_in, channel_out in self.block_channel_sizes])
                                             
        # components for tail end of network
        self.global_avg_pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.fully_connected = nn.Linear(128, 10)
        
        # if gpu is present, use it
        self.device = "cpu"
        self.cuda_enabled = torch.cuda.is_available()
        if self.cuda_enabled:
            self.device = "cuda:0"
            self.cuda()
            
    def tail(self, x):
        x = self.global_avg_pooling_layer(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fully_connected(x)
        return x
    
    def num_flat_features(self, x):
        return reduce(multiply, x.size()[1:], 1)
    
    def forward(self, x):
        return self.tail(self.layers(x))
    
class Net(Network):
    def __init__(self):
        super(Net, self).__init__(BasicBlock)
        
class ResNet(Network):
    def __init__(self):
        super(ResNet, self).__init__(ResNetBasicBlock)
        
def update_progress(progress, acc):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0.0
    if progress < 0:
        progress = 0.0
    if progress >= 1:
        progress = 1.0
    block = int(round(bar_length * progress))
    clear_output(wait=True)
    text = acc
    text += "\nProgress: [{0}] {1:.1f}%".format("#" * block + "-" * (bar_length - block), progress * 100)
    print(text)
import math
import sklearn.metrics as m
def train_network_optim(dataloader, network, criterion, optimizer, scheduler, batch_size,
                        batches=-1, target_num_batches=-1, processed_batches=-1):
    if target_num_batches < 0:
        target_num_batches = math.ceil(
            len(dataloader)/batch_size)
    if batches < 0:
        batches = target_num_batches
    if processed_batches < 0:
        processed_batches = 0
    acc = f"Evaluating with num_batches/target num of batches:" +\
        f"{batches} / {target_num_batches}"
    epoch = 0
    while processed_batches < batches:
        running_loss = 0.0
        for i, data in enumerate(random.shuffle(dataloader), 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            processed_batches += 1
            running_loss += loss.item()
            update_progress(float(processed_batches)/batches, acc)
            if processed_batches % math.ceil(batches/4) == math.ceil(batches/4) - 1:
                acc += "\n[%d, %5d] loss: %.3f\n" % \
                    (epoch + 1, i + 1, running_loss / math.ceil(batches/4))
                running_loss = 0.0
            if processed_batches > batches:
                break
        epoch += 1
        scheduler.step()
    print("Finished Training")
    update_progress(1, acc)
    return processed_batches
def evaluate_network_opt(trainloader, testloader, network_class, batch_size, step_size=20, epochs_desired=-1):
    batches_in_one_epoch = math.ceil(
        len(trainloader)/batch_size)
    if epochs_desired < 0: epochs_desired = 1
    batch_num_target = batches_in_one_epoch * epochs_desired
    performance_when_varying_input_size = {
        num_batches_to_train: None for \
        num_batches_to_train in range(1, batch_num_target, step_size + 1)
    }
    network = network_class()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=0.0001)
    def decay_lr(epoch_num):
        return 0.1 if epoch_num in [32_000, 48_000] else 1
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, decay_lr)
    batches_done = -1
    for num_batches_to_train in range(1, batch_num_target, step_size + 1):
        network.train()
        batches_done = train_network_optim(trainloader, network, criterion, optimizer, scheduler, batch_size,
                                           num_batches_to_train, batch_num_target, batches_done)
        
        network.eval()
        network_outputs = []
        print(f"Evaluating with num_batches/target number of batches:" +\
              f"{num_batches_to_train} / {batch_num_target}")
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                network_outputs.append({"output_score": network(images).to("cpu"),
                                        "actual_labels": labels})
        
        y_true = {label: [] for label in range(0,10)}
        y_score = {label: [] for label in range(0,10)}
        print(f"Calculating AUC by class for num_batches/target number of batches:" +\
              f"{num_batches_to_train} / {batch_num_target}")
        for d in network_outputs:
            for label in range(0,10):
                score, predicted = torch.max(d["output_score"], 1)
                y_true[label] = [*y_true[label], *(d["actual_labels"] == label)]
                y_score[label] = [*y_score[label], *(d["output_score"][:, label])]
        auc_by_label = {label: [] for label in range(1,11)}
        for label in range(0, 10):
            fpr, tpr, thresholds = m.roc_curve(y_true[label], y_score[label])
            auc_by_label[label] = m.auc(fpr, tpr)
        performance_when_varying_input_size[num_batches_to_train] = \
            auc_by_label
    return performance_when_varying_input_size, network
            