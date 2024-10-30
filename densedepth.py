import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class DepthModel(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthModel, self).__init__()
        
        # Using ResNet34 for faster inference while maintaining quality
        encoder = models.resnet34(weights='IMAGENET1K_V1')
        
        # Encoder
        self.first_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1  # 64
        self.layer2 = encoder.layer2  # 128
        self.layer3 = encoder.layer3  # 256
        self.layer4 = encoder.layer4  # 512
        
        # Decoder
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Decoder with skip connections
        x = self.conv1(x4)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = self.conv2(x + x3)  # Skip connection
        x = self.bn3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = self.conv3(x + x2)  # Skip connection
        x = self.bn4(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = self.conv4(x + x1)  # Skip connection
        x = self.bn5(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = self.conv5(x)
        x = torch.sigmoid(x)  # Normalize to [0,1]
        
        return x

class NYUDepthDataset(Dataset):
    def __init__(self, rgb_paths, depth_paths, transform=None):
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.transform = transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_image = Image.open(self.rgb_paths[idx]).convert('RGB')
        depth_image = Image.open(self.depth_paths[idx])
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = transforms.ToTensor()(depth_image)
        
        return rgb_image, depth_image

class DepthEstimator:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = DepthModel().to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded weights from {model_path}")
        
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.target_size = (480, 640)

    def train(self, train_loader, epochs=10, learning_rate=0.001):
        """Train the model on depth dataset"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (rgb, depth) in enumerate(train_loader):
                rgb = rgb.to(self.device)
                depth = depth.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(rgb)
                loss = criterion(outputs, depth)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if i % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {loss.item():.4f}')
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    def predict_depth(self, frame):
        """Predict depth from a single frame"""
        try:
            with torch.no_grad():
                # Preprocess
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                # Predict
                depth = self.model(input_tensor)
                
                # Post-process
                depth = depth.squeeze().cpu().numpy()
                
                # Normalize and enhance visualization
                depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
                
                # Resize to match input frame
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
                
                return depth
                
        except Exception as e:
            print(f"Error in depth prediction: {str(e)}")
            raise

    def start_live_depth(self, source=0):
        """Start live depth estimation"""
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f"Could not open video source {source}")

            print("Starting live depth estimation...")
            print("Press 'q' to quit")
            print("Press 's' to save current frame and depth map")
            
            prev_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    # Calculate FPS
                    current_time = time.time()
                    fps = 1 / (current_time - prev_time)
                    prev_time = current_time
                    
                    # Get depth map
                    depth_map = self.predict_depth(frame)
                    
                    # Add FPS counter
                    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display side by side
                    combined = np.hstack((frame, depth_map))
                    cv2.imshow('Original | Depth (Press q to quit, s to save)', combined)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = int(time.time())
                        cv2.imwrite(f'frame_{timestamp}.jpg', frame)
                        cv2.imwrite(f'depth_{timestamp}.jpg', depth_map)
                        print(f"Saved frame and depth map")

                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    continue

        finally:
            cap.release()
            cv2.destroyAllWindows()

def train_model():
    """Example of how to train the model"""
    # Setup dataset paths (replace with your dataset paths)
    rgb_paths = ['path/to/rgb/images']  # List of RGB image paths
    depth_paths = ['path/to/depth/images']  # List of corresponding depth image paths
    
    # Create dataset and dataloader
    dataset = NYUDepthDataset(
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        transform=transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    )
    
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize and train model
    estimator = DepthEstimator()
    estimator.train(train_loader, epochs=10)
    
    # Save trained model
    torch.save(estimator.model.state_dict(), 'depth_model.pth')

if __name__ == "__main__":
    try:
        # For training:
        # train_model()
        
        # For inference:
        estimator = DepthEstimator()  # Add model_path='depth_model.pth' to use trained weights
        estimator.start_live_depth(0)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise