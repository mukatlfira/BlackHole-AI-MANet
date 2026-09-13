# ===================================================================
# ALL-IN-ONE SCRIPT FOR BLACK HOLE AI RESEARCH (ROBUSTNESS TEST)
# Fully in English for academic and research purposes.
# ===================================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# -------------------- 1. CONFIGURATION --------------------
class Config:
    IMG_SIZE = 64
    NUM_TRAIN = 1000
    NUM_VAL = 200
    NUM_TEST = 200
    NOISE_LEVEL = 0.05
    NUM_PARAMS = 3
    INPUT_CHANNELS = 1
    HIDDEN_DIM = 128
    BATCH_SIZE = 32
    EPOCHS = 10 # Reduced for faster robustness testing
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_SAVE_PATH = "./results/models/best_model.pth"
    FIGURE_SAVE_PATH = "./results/figures/"

# -------------------- 2. DATA GENERATOR --------------------
class BlackHoleSimulator:
    def __init__(self, img_size=64):
        self.img_size = img_size
        self.x = np.linspace(-1, 1, img_size)
        self.y = np.linspace(-1, 1, img_size)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

    def generate(self, spin, inclination, magnetization, noise_std=0.05):
        r = np.sqrt(self.xx**2 + self.yy**2)

        # Photon ring brightness profile (influenced by magnetization)
        brightness_boost = 1.0 + 0.5 * magnetization
        shadow_radius = 0.5 + 0.4 * (1 - spin)
        sharpness = 15 + 15 * (1 - spin)
        ring = brightness_boost / (1 + np.exp(sharpness * (r - shadow_radius)))

        # Doppler asymmetry (Relativistic beaming)
        asymmetry = 1 + 0.5 * spin * inclination * self.xx / (np.abs(self.xx) + 0.05)
        image = ring * asymmetry

        # Inner shadow mask
        inner_mask = r < (shadow_radius * 0.7)
        image[inner_mask] = image[inner_mask] * 0.1 + 0.05

        image = np.clip(image, 0, 1)

        # Add observation noise
        noise = np.random.normal(0, noise_std, (self.img_size, self.img_size))
        image = np.clip(image + noise, 0, 1)

        return image.astype(np.float32), np.array([spin, inclination, magnetization], dtype=np.float32)

    def generate_dataset(self, num_samples, noise_std=0.05):
        images, params = [], []
        for _ in range(num_samples):
            img, p = self.generate(np.random.uniform(0.1,0.95),
                                   np.random.uniform(0.1,0.95),
                                   np.random.uniform(0.1,0.95),
                                   noise_std)
            images.append(img)
            params.append(p)
        return np.stack(images, axis=0), np.stack(params, axis=0)

# -------------------- 3. DATASET --------------------
class BlackHoleDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.expand_dims(self.images[idx].astype(np.float32), axis=0)
        return torch.from_numpy(img), torch.from_numpy(self.labels[idx].astype(np.float32))

# -------------------- 4. MODELS --------------------
class MultiScaleAdaptiveNet(nn.Module):
    """
    Proposed Multi-Scale Adaptive Network (MANet).
    Captures physics at different scales.
    """
    def __init__(self, input_channels=1, num_params=3, hidden_dim=128):
        super().__init__()
        # Branch 1: Small receptive field (local turbulence)
        self.branch1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        # Branch 2: Medium receptive field (photon ring structure)
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU()
        )
        # Branch 3: Large receptive field (global shadow deformation)
        self.branch3 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 7, padding=3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 7, padding=3), nn.BatchNorm2d(32), nn.ReLU()
        )
        # Fusion and Regression Head
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(96, hidden_dim, 1), nn.BatchNorm2d(hidden_dim), nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_params)
        )

    def forward(self, x):
        concat = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        return self.regressor(self.fusion_conv(concat))

# -------------------- 5. TRAINER --------------------
class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)
                total_loss += self.criterion(self.model(images), labels).item()
        return total_loss / len(self.val_loader)

    def run(self):
        for epoch in range(1, self.config.EPOCHS + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                os.makedirs(os.path.dirname(self.config.MODEL_SAVE_PATH), exist_ok=True)
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)

# -------------------- 6. EVALUATOR --------------------
class Evaluator:
    def __init__(self, model, test_loader, config):
        self.model = model.to(config.DEVICE)
        self.test_loader = test_loader
        self.config = config

    def load_best_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.config.DEVICE))

    def evaluate(self):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for images, lbls in self.test_loader:
                images = images.to(self.config.DEVICE)
                preds.append(self.model(images).cpu().numpy())
                labels.append(lbls.numpy())
        preds = np.vstack(preds)
        labels = np.vstack(labels)

        mse = mean_squared_error(labels, preds)
        mae = mean_absolute_error(labels, preds)
        r2 = r2_score(labels, preds)
        return mse, mae, r2

# -------------------- 7. ROBUSTNESS TEST EXECUTION --------------------
if __name__ == "__main__":
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    print("="*60)
    print("STARTING ROBUSTNESS TEST")
    print("="*60)

    for noise in noise_levels:
        print(f"\n--- Training MANet with Noise Level: {noise} ---")

        # Generate data
        sim = BlackHoleSimulator(Config.IMG_SIZE)
        train_img, train_lbl = sim.generate_dataset(Config.NUM_TRAIN, noise)
        val_img, val_lbl = sim.generate_dataset(Config.NUM_VAL, noise)
        test_img, test_lbl = sim.generate_dataset(Config.NUM_TEST, noise)

        # DataLoaders
        train_loader = DataLoader(BlackHoleDataset(train_img, train_lbl), batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(BlackHoleDataset(val_img, val_lbl), batch_size=Config.BATCH_SIZE)
        test_loader = DataLoader(BlackHoleDataset(test_img, test_lbl), batch_size=Config.BATCH_SIZE)

        # Train
        model = MultiScaleAdaptiveNet()
        trainer = Trainer(model, train_loader, val_loader, Config)
        trainer.run()

        # Evaluate
        evaluator = Evaluator(MultiScaleAdaptiveNet(), test_loader, Config)
        evaluator.load_best_model(Config.MODEL_SAVE_PATH)
        mse, mae, r2 = evaluator.evaluate()

        print(f"RESULTS FOR NOISE LEVEL {noise}:")
        print(f"  Test MSE: {mse:.6f}")
        print(f"  Test MAE: {mae:.6f}")
        print(f"  Test R2 : {r2:.4f}")

    print("\n" + "="*60)
    print("ROBUSTNESS TEST COMPLETE")
    print("="*60)
