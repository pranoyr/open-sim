import argparse
import torch
from torch.utils.data import DataLoader
from open_sim.datasets.dataset import AlohaBabblingDataset
from open_sim.models.mimic_video import create_mimic_video_idm

def train(epochs=2):
    print("Initializing MimicVideo IDM...")
    model = create_mimic_video_idm()
    
    # Dataset and Dataloader
    dataset = AlohaBabblingDataset(chunk_len=15, img_size=32)
    if len(dataset) == 0:
        print("No dataset found in babbling-dataset/")
        return
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    print(f"Starting training on {len(dataset)} episodes...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for video, actions, joint_state in dataloader:
            optimizer.zero_grad()
            
            # ALOHA datasets use a placeholder prompt since it's just babbling
            prompts = ['motor babbling'] * video.shape[0]
            
            loss = model(
                prompts=prompts,
                video=video,
                actions=actions,
                joint_state=joint_state
            )
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.4f}")
        
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    
    train(epochs=args.epochs)
