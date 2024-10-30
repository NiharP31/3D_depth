import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

class LiveDepthEstimator:
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

    def estimate_depth(self, frame):
        """
        Estimate depth from a video frame
        Args:
            frame: Input frame from video
        Returns:
            colored_depth: Colored visualization of depth map
        """
        try:
            # Convert frame to RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
            # Convert to BGR for OpenCV display
            colored_depth = (colored_depth[:, :, :3] * 255).astype(np.uint8)
            colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR)

            return colored_depth
            
        except Exception as e:
            print(f"Error during depth estimation: {str(e)}")
            raise

    def start_live_depth(self, source=0):
        """
        Start live depth estimation from webcam or video file
        Args:
            source: Camera index (usually 0 for built-in webcam) or video file path
        """
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f"Could not open video source {source}")

            print("Starting live depth estimation... Press 'q' to quit")

            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Get depth map
                depth_colored = self.estimate_depth(frame)

                # Resize depth map to match original frame size
                depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))

                # Display original and depth side by side
                combined = np.hstack((frame, depth_colored))
                
                # Display result
                cv2.imshow('Live Depth Estimation (Press q to quit)', combined)

                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error in live depth estimation: {str(e)}")
            raise
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Initialize depth estimator
        print("Initializing depth estimator...")
        depth_estimator = LiveDepthEstimator()
        
        # Start live depth estimation
        # Use 0 for webcam, or provide a video file path
        depth_estimator.start_live_depth(0)  # Change to video path if needed
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")