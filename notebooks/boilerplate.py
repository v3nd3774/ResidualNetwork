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
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n = 7
        self.convs_32 = [
            (nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16)) \
             for _ in range(2 * self.n - 2)]
        self.convs_32 = [(
            nn.Conv2d(3,16,3,padding=1), nn.BatchNorm2d(16)
        ),*self.convs_32]
        self.convs_32 = [*self.convs_32, (
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32)
        )]
        self.convs_32_bn = nn.ModuleList([bn for _, bn in self.convs_32])
        self.convs_32 = nn.ModuleList([conv for conv, _ in self.convs_32])
        
        self.convs_16 = [
            (nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32)) \
             for _ in range(2 * self.n - 2)]
        self.convs_16 = [*self.convs_16, (
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64)
        )]
        # put into modulelist for cuda functionality
        self.convs_16_bn = nn.ModuleList([bn for _, bn in self.convs_16])
        self.convs_16 = nn.ModuleList([conv for conv, _ in self.convs_16])
        
        self.convs_8 = [(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)) \
                        for _ in range(2 * self.n - 1)]
        self.convs_8 = [*self.convs_8, (
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128)
        )]
        self.convs_8_bn = nn.ModuleList([bn for _, bn in self.convs_8])
        self.convs_8 = nn.ModuleList([conv for conv, _ in self.convs_8])
        
        self.global_avg_pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.fully_connected = nn.Linear(128, 10)
        self.device = "cpu"
        self.cuda_enabled = torch.cuda.is_available()
        if self.cuda_enabled:
            self.device = "cuda:0"
            self.cuda()
    def num_flat_features(self, x):
        return reduce(multiply, x.size()[1:], 1)
    def tail_end_of_network(self, x):
        x = self.global_avg_pooling_layer(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fully_connected(x)
        return x
    def forward(self, x):
        for conv, bn in zip([*self.convs_32, *self.convs_16, *self.convs_8],
                            [*self.convs_32_bn, *self.convs_16_bn, *self.convs_8_bn]):
            x = F.relu(bn(conv(x)))
        return self.tail_end_of_network(x)
class ResNet(Net):
    def __init__(self):
        super(ResNet, self).__init__()
    def forward(self, x):
        first_conv, first_bn = self.convs_32[0], self.convs_32_bn[0]
        x = F.relu(first_bn(first_conv(x)))
        for idx, (conv, bn) in enumerate(zip(
            [*self.convs_32[1:], *self.convs_16, *self.convs_8],
            [*self.convs_32_bn[1:], *self.convs_16_bn, *self.convs_8_bn])):
            if idx % 2 == 0:
                identity = x
                x = F.relu(bn(conv(x)))
            else:
                if bn(conv(x)).shape == identity.shape:
                    x = F.relu(bn(conv(x) + identity))
                else:
                    x = F.relu(bn(conv(x)))
        return self.tail_end_of_network(x)
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
def train_network_optim(dataloader, network, criterion, optimizer, scheduler,
                        batches=-1, target_num_batches=-1, processed_batches=-1):
    if target_num_batches < 0:
        target_num_batches = math.ceil(
            len(dataloader.dataset)/dataloader.batch_size)
    if batches < 0:
        batches = target_num_batches
    if processed_batches < 0:
        processed_batches = 0
    acc = f"Evaluating with num_batches/target num of batches:" +\
        f"{batches} / {target_num_batches}"
    epoch = 0
    while processed_batches < batches:
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = list(map(lambda data: data.to(network.device), data)) \
                             if network.cuda_enabled else data
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            processed_batches += 1
            running_loss += loss.item()
            update_progress(float(processed_batches)/batches, acc)
            if processed_batches % math.ceil(batches/4) == math.ceil(batches/4) - 1:
                acc += "[%d, %5d] loss: %.3f\n" % \
                    (epoch + 1, i + 1, running_loss / math.ceil(batches/4))
                running_loss = 0.0
            if processed_batches > batches:
                break
        epoch += 1
        scheduler.step()
    print("Finished Training")
    update_progress(1, acc)
    return processed_batches
def evaluate_network_opt(trainloader, testloader, network_class, step_size=20, epochs_desired=-1):
    batches_in_one_epoch = math.ceil(
        len(trainloader.dataset)/trainloader.batch_size)
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
        batches_done = train_network_optim(trainloader, network, criterion, optimizer, scheduler,
                                           num_batches_to_train, batch_num_target, batches_done)
        
        network.eval()
        network_outputs = []
        print(f"Evaluating with num_batches/target number of batches:" +\
              f"{num_batches_to_train} / {batch_num_target}")
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                if network.cuda_enabled:
                    images = images.to(network.device)
                network_outputs.append({"output_score": network(images),
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
        return auc_by_label, network
            