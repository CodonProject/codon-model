import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from codon.base  import *
from codon.block import (
    MLP,
    GatedMultimodalUnit,
    LowRankFusion,
    FiLM, FiLMOutput
)
from codon.block.adanorm import AdaLayerNorm

from codon.utils.seed import seed_everything
from codon.utils.eval import ConfusionMap
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

seed_everything(seed=42, verbose=False)


class FastAttn(BasicModel):
    def __init__(self, way: str = 'low'):
        super().__init__()
        self.way = way
        self.linear = nn.Linear(28*28, 14)

        if way == 'low':
            self.fusion = LowRankFusion(
                in_features=[28*28, 14],
                out_features=28*28,
                rank=16
            )
        elif way == 'gated':
            self.fusion = GatedMultimodalUnit(
                in_features=[28*28, 14], 
                out_features=28*28
            )
        elif way == 'film':
            self.film = FiLM(
                in_features=28*28,
                cond_features=14,
                use_gate=True
            )
        elif way == 'adanorm':
            self.fusion = AdaLayerNorm(
                features_dim=28*28,
                embedding_dim=14
            )
    
    def forward(self, x: torch.Tensor):
        attn: torch.Tensor = self.linear(x)

        if self.way == 'film': 
            film_output: FiLMOutput = self.film(x, attn)
            return film_output.gated_output
        
        elif self.way == 'adanorm':
            return self.fusion(x, attn)
        
        else: 
            return self.fusion([x, attn])


class VisionCore(BasicModel):
    def __init__(self, way: str = None):
        super().__init__()
        if isinstance(way, str) and way == '': 
            way = None
        self.way = way

        self.f = nn.Flatten()
        self.v1 = MLP(28*28, 512, 128)
        self.v2 = MLP(128, 64, 10)
        
        if way is not None:
            self.attn = FastAttn(way=way)

    def forward(self, x):
        x = self.f(x)
        if self.way is not None: 
            x = self.attn(x)
        return self.v2(self.v1(x))


def prepare_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../dataset/mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../dataset/mnist', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device, writer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        step = epoch * len(loader) + batch_idx
        writer.add_scalar('Loss/train_step', loss.item(), step)
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate_model(model, loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
    epoch_loss = test_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Active Device: {device}')

    train_loader, test_loader = prepare_loader()
    
    ways = ['', 'gated', 'low', 'film', 'adanorm']
    epochs = 10
    
    for way in ways:
        way_tag = way if way != '' else 'baseline_no_attn'
        print(f'Starting configuration: [{way_tag.upper()}]')
        
        model = VisionCore(way=way).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        writer = SummaryWriter(log_dir=f'runs/MNIST_comparison/{way_tag}')
        
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, writer, epoch
            )
            test_loss, test_acc = evaluate_model(
                model, test_loader, criterion, device
            )
            
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Loss/test_epoch', test_loss, epoch)
            writer.add_scalar('Accuracy/train_epoch', train_acc, epoch)
            writer.add_scalar('Accuracy/test_epoch', test_acc, epoch)
            
            print(f'Epoch {epoch+1:02d}/{epochs:02d} | '
                  f'Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | '
                  f'Test Loss: {test_loss:.4f} (Acc: {test_acc:.2f}%)')
        
        print(f'Generating Confusion Matrix for [{way_tag}]...')
        confusion_map = ConfusionMap(10, test_loader)
        
        analysis = confusion_map.analyse(model)
        fig: plt.Figure = analysis.fig
        
        writer.add_figure('Evaluation/ConfusionMatrix', fig, global_step=epochs)
        
        plt.close(fig)
        writer.close()
        
        print(f'Configuration [{way_tag}] execution complete and logged.')