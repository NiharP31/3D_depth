import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class DepthEstimator:
    def __init__(self):
        """Initialize the depth estimation model"""
        try:
            # Load MiDaS model
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def estimate_depth(self, image_path):
        """
        Estimate depth from a single image
        Args:
            image_path: Path to input image
        Returns:
            depth_map: Normalized depth map
            colored_depth: Depth map converted to colored visualization
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Transform input for model
            input_batch = self.transform(img).to(self.device)

            # Prediction and post-processing
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()
            
            # Normalize depth map
            depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            # Create colored depth map
            colored_depth = plt.cm.viridis(depth_map)
            colored_depth = (colored_depth * 255).astype(np.uint8)

            return depth_map, colored_depth
            
        except Exception as e:
            print(f"Error during depth estimation: {str(e)}")
            raise

    def visualize_results(self, image_path, save_path=None):
        """
        Visualize original image alongside depth map
        Args:
            image_path: Path to input image
            save_path: Optional path to save visualization
        """
        try:
            # Get depth estimation
            depth_map, colored_depth = self.estimate_depth(image_path)
            
            # Read original image
            original = cv2.imread(image_path)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            # Create visualization
            plt.figure(figsize=(12, 5))
            
            plt.subplot(121)
            plt.imshow(original)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(colored_depth)
            plt.title('Depth Map')
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path)
                print(f"Visualization saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Initialize depth estimator
        print("Initializing depth estimator...")
        depth_estimator = DepthEstimator()
        
        # Define image path - replace with your image path
        image_path = "image.png"  # Put your image path here
        
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Please place an image at {image_path} or modify the path in the code.")
        
        # Process and visualize results
        print(f"Processing image: {image_path}")
        depth_estimator.visualize_results(image_path, "image.png")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")