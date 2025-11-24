"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

print("Time to train")

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from homework.datasets.road_dataset import load_data
from homework import models

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2025,
    **kwargs,
):
    
    # 1. Setup Device for Colab
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Colab Using Device: {device}")

    # 2. Load data, MLP no need the images so the transform is state_only
    train_loader = load_data(
        "drive_data/train",
        transform_pipeline="state_only",
        batch_size=batch_size,
        shuffle=True,
    )

    # 3. Create Model, Optimizer, and Loss Function
    model = models.load_model(model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    # 4. Training Loop
    for epoch in range(num_epochs):
        
        model.train()

        total_loss = 0.0

        for batch in train_loader:

            # Move data to GPU Device
            track_left = batch["track_left"].to(device)             # Left Track
            track_right = batch["track_right"].to(device)           # Right Track
            waypoints = batch["waypoints"].to(device)               # Ground TruthLabels
            waypoints_mask = batch["waypoints_mask"].to(device)     # Mask for validating waypoints

            # Forward Pass
            pred_waypoints = model(track_left, track_right)

            # Compute Loss in valid waypoints only by multiplying the mask
            mask = waypoints_mask[..., None]
            loss = loss_fn(pred_waypoints * mask, waypoints * mask)

            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update total loss
            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")

    # Save Model
    models.save_model(model)
    print(f"Model saved to {model_name}.th")


# 5. Training Script
if __name__ == "__main__":
    
    # Parse Arguments for tell the train function use what model, num epochs, lr, batch size, seed
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mlp_planner")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2025)

    args = parser.parse_args()

    # Train Model
    train(**vars(args))
    





    
