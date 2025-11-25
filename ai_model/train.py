import torch
import torch.optim as optim
from model import get_model
import os

def train():
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print("Model initialized.")
    print(model)

    # Dummy training loop
    # In a real scenario, you would load the DataLoader here
    # for epoch in range(10):
    #     for batch in dataloader:
    #         optimizer.zero_grad()
    #         output = model(batch.image.to(device), batch.graph.to(device))
    #         loss = criterion(output, batch.target.to(device))
    #         loss.backward()
    #         optimizer.step()
    
    # Save model
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'saved_models/chemnet_vision.pth')
    print("Model saved.")

if __name__ == "__main__":
    train()
