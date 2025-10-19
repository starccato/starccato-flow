import torch
# from starccato_flow.nn import flow
from starccato_flow.nn.maf import MaskedAutoregressiveFlow
# from starccato_flow.nn.maf_2 import MAF
from torch.utils.data import DataLoader, TensorDataset
from starccato_flow.data.ccsn_data import CCSNData
from starccato_flow.data.toy_data import ToyData

import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from starccato_flow.utils.defaults import DEVICE, BATCH_SIZE

def train_flow(epochs=50, batch_size=BATCH_SIZE, lr=1e-3):
    flow = MaskedAutoregressiveFlow(n_layers=10, dim=2, hidden_dims=[8, 8]).to(DEVICE)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)

    # Generate 2D "two moons" dataset
    data, _ = make_moons(n_samples=2000, noise=0.1)
    data = torch.tensor(data, dtype=torch.float32)
    # data = (data - data.mean(0)) / data.std(0)

    plt.figure(figsize=(4,4))
    plt.scatter(data[:,0], data[:,1], s=5)
    plt.title("Toy 2D Data (Two Moons)")
    plt.show()
    
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        flow.train( )
        total_nll = 0.0
        for batch, in loader:
            batch = batch.to(DEVICE)
            _, log_px = flow(batch)
            loss = -log_px.mean()  # negative log-likelihood
            opt.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
            opt.step()
            total_nll += loss.item() * batch.size(0)

        # Optional: Print intermediate results
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} Batch NLL: {loss.item():.4f}")
            flow.eval()
            with torch.no_grad():
                samples = flow.sample(1000, device=DEVICE).detach().cpu()

                plt.figure(figsize=(4,4))
                plt.scatter(samples[:,0], samples[:,1], s=5, c='orange')
                plt.title("Samples from trained MAF")
                plt.show()

        print(f"Epoch {epoch+1}/{epochs} NLL: {total_nll / len(data):.4f}")