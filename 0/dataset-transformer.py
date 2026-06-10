import torch
import torchvision
import numpy as np
import os

# Load MNIST dataset with initial transform
dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

class DatasetTransformer(torch.utils.data.Dataset):
    """Generic dataset wrapper to apply additional transformations."""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    
class SpeechCommandsDataset(torch.utils.data.Dataset):
    """Custom Speech Commands dataset wrapper."""
    def __init__(self, data_dir="./data/speech_commands", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = {}
        self.label_idx = 0
        
        # Load synthetic speech data if directory doesn't exist
        if not os.path.exists(data_dir):
            self._create_synthetic_data()
        
        # Load all samples
        self._load_samples()
    
    def _create_synthetic_data(self):
        """Create synthetic speech command data."""
        os.makedirs(self.data_dir, exist_ok=True)
        commands = ["yes", "no", "up", "down", "left", "right"]
        
        for cmd in commands:
            cmd_dir = os.path.join(self.data_dir, cmd)
            os.makedirs(cmd_dir, exist_ok=True)
            
            # Create 10 synthetic samples per command
            for i in range(10):
                # Generate random audio-like data (1D tensor)
                audio_data = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
                np.save(os.path.join(cmd_dir, f"sample_{i}.npy"), audio_data)
    
    def _load_samples(self):
        """Load all samples from directory."""
        for label_name in sorted(os.listdir(self.data_dir)):
            label_dir = os.path.join(self.data_dir, label_name)
            if os.path.isdir(label_dir):
                self.labels[label_name] = self.label_idx
                
                for filename in sorted(os.listdir(label_dir)):
                    if filename.endswith(".npy"):
                        self.samples.append((os.path.join(label_dir, filename), self.label_idx))
                
                self.label_idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        audio = np.load(audio_path)
        audio = torch.from_numpy(audio).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            audio = self.transform(audio)
        
        return audio, label

    
class HindiSpeechCommandsDataset(torch.utils.data.Dataset):
    """Custom Hindi Speech Commands dataset wrapper."""
    def __init__(self, data_dir="./data/hindi_speech_commands", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = {}
        self.label_idx = 0
        
        # Load synthetic Hindi speech data if directory doesn't exist
        if not os.path.exists(data_dir):
            self._create_synthetic_data()
        
        # Load all samples
        self._load_samples()
    
    def _create_synthetic_data(self):
        """Create synthetic Hindi speech command data."""
        os.makedirs(self.data_dir, exist_ok=True)
        commands = ["हां", "नहीं", "ऊपर", "नीचे", "बाएं", "दाएं"]  # yes, no, up, down, left, right in Hindi
        
        for cmd in commands:
            cmd_dir = os.path.join(self.data_dir, cmd)
            os.makedirs(cmd_dir, exist_ok=True)
            
            # Create 10 synthetic samples per command
            for i in range(10):
                audio_data = np.random.randn(16000).astype(np.float32)
                np.save(os.path.join(cmd_dir, f"sample_{i}.npy"), audio_data)
    
    def _load_samples(self):
        """Load all samples from directory."""
        for label_name in sorted(os.listdir(self.data_dir)):
            label_dir = os.path.join(self.data_dir, label_name)
            if os.path.isdir(label_dir):
                self.labels[label_name] = self.label_idx
                
                for filename in sorted(os.listdir(label_dir)):
                    if filename.endswith(".npy"):
                        self.samples.append((os.path.join(label_dir, filename), self.label_idx))
                
                self.label_idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        audio = np.load(audio_path)
        audio = torch.from_numpy(audio).unsqueeze(0)
        
        if self.transform:
            audio = self.transform(audio)
        
        return audio, label


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Testing MNIST Dataset with Transform")
    print("=" * 60)
    
    # MNIST with transform
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
    ])
    
    transformed_dataset = DatasetTransformer(dataset, transform)
    print(f"Original MNIST size: {len(dataset)}")
    print(f"Transformed MNIST size: {len(transformed_dataset)}")
    x, y = transformed_dataset[0]
    print(f"MNIST Sample shape: {x.shape}, Label: {y}\n")
    
    print("=" * 60)
    print("Testing Speech Commands Dataset")
    print("=" * 60)
    
    # Speech Commands
    speech_dataset = SpeechCommandsDataset()
    print(f"Speech Commands dataset size: {len(speech_dataset)}")
    print(f"Labels: {speech_dataset.labels}")
    if len(speech_dataset) > 0:
        audio, label = speech_dataset[0]
        print(f"Audio shape: {audio.shape}, Label: {label}\n")
    
    print("=" * 60)
    print("Testing Hindi Speech Commands Dataset")
    print("=" * 60)
    
    # Hindi Speech Commands
    hindi_dataset = HindiSpeechCommandsDataset()
    print(f"Hindi Speech Commands dataset size: {len(hindi_dataset)}")
    print(f"Labels: {hindi_dataset.labels}")
    if len(hindi_dataset) > 0:
        audio, label = hindi_dataset[0]
        print(f"Audio shape: {audio.shape}, Label: {label}")

