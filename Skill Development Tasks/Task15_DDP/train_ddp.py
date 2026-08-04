import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ----------------------------------------------------
# Simple Neural Network
# ----------------------------------------------------
class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

# ----------------------------------------------------
# Initialize Distributed Process
# ----------------------------------------------------
def setup(rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )

# ----------------------------------------------------
# Destroy Process
# ----------------------------------------------------
def cleanup():
    dist.destroy_process_group()

# ----------------------------------------------------
# Training Function
# ----------------------------------------------------
def train(rank, world_size):

    print(f"Starting Process {rank}")

    setup(rank, world_size)

    torch.manual_seed(42)

    # Synthetic Dataset
    X = torch.randn(5000, 20)
    y = torch.randint(0, 2, (5000,))

    dataset = TensorDataset(X, y)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler
    )

    model = SimpleNet()

    model = DDP(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 5

    for epoch in range(epochs):

        sampler.set_epoch(epoch)

        total_loss = 0

        for inputs, labels in loader:

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Rank {rank} | Epoch {epoch+1}/{epochs} | Loss = {total_loss:.4f}"
        )

    cleanup()

# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():

    world_size = 2

    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()