import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from constants import MODEL_NAME


class InsuranceDataset(Dataset):
    """
    A custom dataset class for our syntheitc insurance documents.
    
    This class helps us prepare our data in a format that the transformer models can use.
    """
    def __init__(self, texts: list[str], labels: list[int], tokenizer: AutoTokenizer, max_length: int = 512) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,      
            padding=True,         
            max_length=max_length,
            return_tensors='pt'   
        )
        self.labels = labels

    def __getitem__(self, idx):
        # Return a single example when requested
        item = {
            key: val[idx] 
            for key, val in self.encodings.items()
        }
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def load_and_prepare_data(file_path: str, test_size: float = 0.2):
    """
    Loads our insurance dataset and prepares it for training.
    
    This function:
    1. Reads the JSON file
    2. Splits data into training and testing sets
    3. Creates dataset objects for both sets
    """
    # First, let's load our data
    print("Loading dataset from:", file_path)
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract texts and labels
    texts = [item['text'] for item in data['dataset']]
    labels = [item['label'] for item in data['dataset']]
    
    # Split the data using sklearn
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, 
        labels,
        test_size=test_size,
        random_state=42  # For reproducibility
    )
    
    print(f"\nDataset split complete:")
    print(f"Training examples: {len(train_texts)}")
    print(f"Testing examples: {len(test_texts)}")
    
    return {
        'train': (train_texts, train_labels),
        'test': (test_texts, test_labels)
    }


def create_datasets(data_splits: dict[str, tuple[list[str], list[int]]], 
                   model_name: str = MODEL_NAME
                   ) -> tuple[AutoTokenizer, InsuranceDataset, InsuranceDataset]:
    """
    Creates PyTorch datasets from our data splits.
    
    This function:
    1. Initializes the tokenizer
    2. Creates training and testing datasets
    """
    # Initialize the tokenizer
    print(f"\nInitializing tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = InsuranceDataset(
        data_splits['train'][0],  # texts
        data_splits['train'][1],  # labels
        tokenizer
    )
    
    test_dataset = InsuranceDataset(
        data_splits['test'][0],   # texts
        data_splits['test'][1],   # labels
        tokenizer
    )
    
    return tokenizer, train_dataset, test_dataset
