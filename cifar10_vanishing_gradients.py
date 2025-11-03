# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 11:06:17 2025

@author: Manisha
"""

# cifar10_vanishing_gradients.py
# PyTorch script: compare deep CNNs with Sigmoid vs ReLU (4 conv layers) 
# on CIFAR-10
# Produces training curves + per-layer gradient norms + optional activation 
# grad heatmaps
#
# Run: python cifar10_vanishing_gradients.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from collections import defaultdict

# -------------------------
# Config / Hyperparameters
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
seed = 42
torch.manual_seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(seed)

batch_size = 128          # reduce if CPU
epochs = 20               # increase if you want stronger convergence
lr = 0.001
save_dir = "cifar_vanish_results"
os.makedirs(save_dir, exist_ok=True)

# -------------------------
# CIFAR-10 transforms
# -------------------------
# Standard CIFAR normalization
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

# -------------------------
# Data loaders
# -------------------------
trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2,
    pin_memory=(device.type=='cuda'))

testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=2, 
    pin_memory=(device.type=='cuda'))

print("Train batches:", len(trainloader), "Test batches:", len(testloader))

# -------------------------
# Model: 4-conv-layer CNN
# -------------------------
class DeepCNN(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        # 4 conv blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   
        # output 32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        # output 64x32x32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        # output 128x32x32
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # output 128x32x32
        self.pool = nn.MaxPool2d(2, 2)  # halves spatial dims

        self.fc = nn.Linear(128 * 4 * 4, 10)  
        # after 3 pools: 32->16->8->4 if pooling after conv1,2,3

        if activation.lower() == 'relu':
            self.act = F.relu
            self.act_name = 'ReLU'
        elif activation.lower() == 'sigmoid':
            self.act = torch.sigmoid
            self.act_name = 'Sigmoid'
        elif activation.lower() == 'tanh':
            self.act = torch.tanh
            self.act_name = 'Tanh'
        else:
            raise ValueError("Unsupported activation")

    def forward(self, x):
        # conv1 -> act -> pool
        x = self.act(self.conv1(x))
        x = self.pool(x)  # 32 -> 16
        # conv2 -> act -> pool
        x = self.act(self.conv2(x))
        x = self.pool(x)  # 16 -> 8
        # conv3 -> act -> pool
        x = self.act(self.conv3(x))
        x = self.pool(x)  # 8 -> 4
        # conv4 -> act (no pool)
        x = self.act(self.conv4(x))  # keep 4x4
        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -------------------------
# Utility functions
# -------------------------
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, 100.0 * correct / total

# record gradient norms helper
def record_grad_norms(model):
    # returns dict: layer_name -> mean(abs(grad)) for weight param
    d = {}
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            if param.grad is None:
                d[name] = 0.0
            else:
                d[name] = param.grad.detach().abs().mean().item()
    return d

# -------------------------
# Train + record gradients per epoch
# -------------------------
def train_and_log(act_name, epochs=20, lr=1e-3):
    print(f"\nTraining model with activation = {act_name}\n")
    model = DeepCNN(activation=act_name.lower()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # storage
    test_losses = []
    test_accs = []
    grad_history = defaultdict(list)  # key: layer name -> list per epoch

    start_time = time.time()
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # after backward, capture per-layer gradient means 
            # (accumulate by batch then avg by epoch)
            grad_norms_batch = record_grad_norms(model)
            # store batch-level grads to aggregate
            for k, v in grad_norms_batch.items():
                grad_history[k].append(v)

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss_epoch = running_loss / total
        train_acc_epoch = 100.0 * correct / total
        test_loss_epoch, test_acc_epoch = evaluate(model, testloader, device)
        test_losses.append(test_loss_epoch)
        test_accs.append(test_acc_epoch)

        # compute per-layer mean gradient for this epoch from recorded batch
        # entries
        epoch_grad_means = {}
        # To be robust: average all stored batch grads for this epoch for 
        # each layer
        for param_name in sorted([k for k in grad_history.keys()]):
            # we store many batches across epochs; approximate epoch-level 
            # by taking the most recent N_batch entries
            # But easier: compute mean over entire grad_history list for now 
            # (will reflect activity)
            epoch_grad_means[param_name] = np.mean(
                grad_history[param_name]
                [-len(trainloader):]) if len(grad_history[param_name]
                                             )>=len(trainloader) else np.mean(
                                                 grad_history[param_name])

        print(f"{act_name} Epoch {epoch}/{epochs}  TrainLoss: "
              f"{train_loss_epoch:.4f}  TrainAcc: {train_acc_epoch:.2f}% "
              f"TestLoss: {test_loss_epoch:.4f}  TestAcc: {test_acc_epoch:.2f}%")
        # flush to disk logs occasionally
    end_time = time.time()
    total_time = end_time - start_time

    # compute final per-layer sequence: for plotting per-epoch we will 
    # compute mean over each epoch by slicing
    # Build layer->list_per_epoch by grouping grad_history per batches per 
    # epoch count
    layer_names = sorted([k for k in grad_history.keys()])
    grad_history_by_epoch = {k: [] for k in layer_names}
    batches_per_epoch = len(trainloader)
    # reshape flattened list into epochs (last incomplete epoch uses 
    # remaining batches)
    for k in layer_names:
        flat = grad_history[k]
        # If too few elements (rare), just pad with zeros
        for e in range(epochs):
            start = e * batches_per_epoch
            end = start + batches_per_epoch
            slice_vals = flat[start:end] if end <= len(flat) else flat[start:len(flat)]
            if len(slice_vals) == 0:
                grad_history_by_epoch[k].append(0.0)
            else:
                grad_history_by_epoch[k].append(float(np.mean(slice_vals)))

    stats = {
        'model': model,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'grad_history_by_epoch': grad_history_by_epoch,
        'time_sec': total_time
    }
    return stats

# -------------------------
# Run experiments (Sigmoid & ReLU)
# -------------------------
# WARNING: This will train two models consecutively. It may be slow on CPU.
results_sig = train_and_log('sigmoid', epochs=epochs, lr=lr)
results_relu = train_and_log('relu', epochs=epochs, lr=lr)

# -------------------------
# Plotting: loss & accuracy
# -------------------------
epochs_range = list(range(1, epochs+1))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, results_sig['test_losses'], 'r-', 
         label='Sigmoid Test Loss')
plt.plot(epochs_range, results_relu['test_losses'], 'b-', 
         label='ReLU Test Loss')
plt.xlabel("Epoch");
plt.ylabel("Loss"); 
plt.title("Test Loss"); 
plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(epochs_range, results_sig['test_accs'], 'r-', label='Sigmoid Test Acc')
plt.plot(epochs_range, results_relu['test_accs'], 'b-', label='ReLU Test Acc')
plt.xlabel("Epoch"); 
plt.ylabel("Accuracy (%)"); 
plt.title("Test Accuracy"); 
plt.legend();
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "loss_acc_comparison.png"), dpi=200)
plt.show()

# -------------------------
# Plot gradient norms per conv layer (mean absolute gradient of weights) per
#  epoch
# -------------------------
# layer key names might look like: conv1.weight, conv2.weight, conv3.weight, 
# conv4.weight
layer_names = sorted(results_sig['grad_history_by_epoch'].keys())
plt.figure(figsize=(12, 6))
for i, layer in enumerate(layer_names):
    plt.subplot(2, 2, i+1)
    plt.plot(epochs_range, results_sig['grad_history_by_epoch'][layer], 
             'r-o', label=f"Sigmoid {layer}")
    plt.plot(epochs_range, results_relu['grad_history_by_epoch'][layer], 
             'b-o', label=f"ReLU {layer}")
    plt.xlabel("Epoch"); plt.ylabel("Mean |grad|"); plt.title(layer)
    plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "grad_norms_per_layer.png"), dpi=200)
plt.show()

# -------------------------
# Activation gradient heatmap for conv1 (optional)
# We'll compute mean abs activation gradient across batch and channels for conv1
# -------------------------
def compute_activation_gradmap(model, loader, device):
    # take one batch from loader
    images, labels = next(iter(loader))
    images = images.to(device); labels = labels.to(device)
    model.zero_grad()
    # forward but ensure activations are retained for gradient (we used 
    # activation functions inline so we can't use retain_grad on tensors 
    # not stored)
    # To capture activation grads we need model to expose activation; we can
    # re-run forward with hooks
    acts = {}
    def save_activation(name):
        def hook(module, inp, out):
            out.retain_grad()
            acts[name] = out
        return hook

    # register hooks on conv1
    handles = []
    handles.append(model.conv1.register_forward_hook(save_activation('conv1')))
    outs = model(images)
    loss = nn.CrossEntropyLoss()(outs, labels)
    loss.backward()
    # conv1 activation gradients
    conv1_act = acts['conv1']  # shape: [B, C, H, W]
    gradmap = conv1_act.grad.detach().abs().mean(dim=1).cpu()  
    # mean over channels -> [B, H, W]
    # cleanup
    for h in handles:
        h.remove()
    return gradmap  # [B, H, W]

# compute heatmap (first sample)
sig_map = compute_activation_gradmap(results_sig['model'], testloader, device)
rel_map = compute_activation_gradmap(results_relu['model'], testloader, device)

# average across batch for a single HxW map
sig_mean_map = sig_map.mean(dim=0).numpy()
rel_mean_map = rel_map.mean(dim=0).numpy()

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(sig_mean_map, cmap='inferno'); 
plt.title("Sigmoid conv1 activation |grad| (mean)"); plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(rel_mean_map, cmap='inferno'); 
plt.title("ReLU conv1 activation |grad| (mean)"); plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "activation_gradmaps_conv1.png"), dpi=200)
plt.show()

print("All results saved to:", save_dir)
