import numpy as np
import evaluate

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from constants import MODEL_NAME

class ModelTrainer:
    """
    Handles the training and evaluation of our insurance document classifier.
    """
    def __init__(self, model_name=MODEL_NAME, num_labels=3):
        self.model_name = model_name
        
        # Initialize our model for fine-tuning
        print(f"Initializing {model_name} for fine-tuning...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
    def compute_metrics(self, pred):
        """
        Calculates metrics for model evaluation.
        This helps us understand how well our model is performing.
        """
        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)
        
        # Calculate accuracy
        accuracy_metric = evaluate.load("accuracy")
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        
        # Calculate F1 (we use 'weighted' average since we might have imbalanced classes)
        f1_metric = evaluate.load("f1")
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        
        # Combine metrics into a single dictionary
        metrics = {
            "accuracy": accuracy["accuracy"],  # Unwrap accuracy from its dictionary
            "f1": f1["f1"]                    # Unwrap f1 from its dictionary
        }
        
        return metrics

    def train(self, train_dataset, test_dataset, output_dir='./results'):
        """
        Fine-tunes the model on our insurance dataset.
        """
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,              # Number of training epochs
            per_device_train_batch_size=8,   # Batch size for training
            per_device_eval_batch_size=8,    # Batch size for evaluation
            warmup_steps=10,                 # Number of warmup steps
            weight_decay=0.05,
            learning_rate=0.0001,
            logging_dir='./logs',            # Directory for storing logs
            logging_steps=5,
            eval_strategy="epoch",     # When to evaluate
            save_strategy="epoch",           # When to save checkpoint
            load_best_model_at_end=True,
            greater_is_better=True,
            metric_for_best_model="eval_f1"       # Use F1 score to determine best model
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics
        )

        print("\nStarting training...")
        
        # Train the model
        train_result = trainer.train()
        
        # Evaluate the model
        eval_result = trainer.evaluate()
        
        print("\nTraining completed!")
        print("\nEvaluation Results:")
        for key, value in eval_result.items():
            print(f"{key}: {value:.4f}")
            
        return trainer, eval_result
