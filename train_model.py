import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# ========== 1) Load dataset ==========
with open("chess_dataset.pkl", "rb") as f:
    data = pickle.load(f)

# collect inputs and move labels
X = np.array([sample["x"] for sample in data])          # shape: (#samples, 64)
moves = [sample["y"] for sample in data]

# encode move strings (e.g. "e2e4", "g1f3") to numbers
le = LabelEncoder()
y = le.fit_transform(moves)                             # shape: (#samples,)

num_classes = len(le.classes_)
print("Total samples:", len(X), " | Number of unique moves:", num_classes)

# ========== 2) Build PyTorch Dataset ==========
class ChessDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = ChessDataset(X, y)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# ========== 3) Build Model (4-layer feedforward) ==========
class ChessNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = ChessNet(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ========== 4) Loss + Optimizer ==========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)

# ========== 5) Training loop ==========
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss {total_loss/len(dataloader):.4f}")

# ========== 6) Save model + label encoder ==========
torch.save(model.state_dict(), "chess_model.pth")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model + label encoder saved!")
