import torch
from starccato_flow.nn.maf import MaskedAutoregressiveFlow
from torch.utils.data import DataLoader, TensorDataset
from starccato_flow.data.ccsn_data import CCSNData
from starccato_flow.data.toy_data import ToyData

import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from starccato_flow.utils.defaults import DEVICE, BATCH_SIZE

def train_flow(epochs=50, batch_size=BATCH_SIZE, lr=1e-3):
    flow = MaskedAutoregressiveFlow(dim=12).to(DEVICE)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)

    # Generate 2D "two moons" dataset
    data, _ = make_moons(n_samples=2000, noise=0.1)
    data = torch.tensor(data, dtype=torch.float32)

    plt.figure(figsize=(4,4))
    plt.scatter(data[:,0], data[:,1], s=5)
    plt.title("Toy 2D Data (Two Moons)")
    plt.show()
    
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_nll = 0.0
        for batch, in loader:
            batch = batch.to(DEVICE)
            _, log_px = flow(batch)
            loss = -log_px.mean()  # negative log-likelihood
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_nll += loss.item() * batch.size(0)
        print(f"Epoch {epoch+1}/{epochs} NLL: {total_nll / len(data):.4f}")
