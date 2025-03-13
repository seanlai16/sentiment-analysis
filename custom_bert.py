import pandas as pd
import numpy as np
import torch
import optuna
from datasets import Dataset
from langdetect import detect
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from datasets import DatasetDict

def load_and_preprocess_data(data_path):
    """Load and preprocess the Amazon reviews dataset."""
    # Create cache directory if it doesn't exist
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if processed data exists in cache
    cache_file = os.path.join(cache_dir, "processed_data.pkl")
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Fill missing values
    data.fillna('', inplace=True)
    
    # Combine title and text
    data['text'] = data['review_title'] + ' ' + data['review_text']
    
    # Create binary labels (0 and 1)
    data['label'] = data['class_index'] - 1
    
    # Cache the processed data
    data.to_pickle(cache_file)
    
    return data

def batch_detect_language(texts, batch_size=1000):
    """Detect language for a batch of texts."""
    cache_dir = ".cache"
    cache_file = os.path.join(cache_dir, "language_detection.npy")
    
    if os.path.exists(cache_file):
        return np.load(cache_file)
    
    is_english = np.zeros(len(texts), dtype=bool)
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        is_english[i:i + batch_size] = [_detect_language(text) for text in batch]
    
    np.save(cache_file, is_english)
    return is_english

def _detect_language(text):
    """Helper function to detect language with error handling."""
    if not isinstance(text, str) or len(text.strip()) < 10:  # Require at least 10 characters
        return False
    
    try:
        return detect(text) == 'en'
    except Exception as e:
        print(f"Warning: Language detection failed for text: {text[:100]}... Error: {str(e)}")
        return False  # Consider non-detectable text as non-English

def prepare_dataset(data):
    """Prepare the dataset for BERT fine-tuning."""
    # Filter for English texts
    data = data[batch_detect_language(data['text'])]
    
    # Create dataset object
    dataset = Dataset.from_dict(data)
    
    # Split into train and test
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    return dataset

def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset."""
    # Check if cached version exists
    cache_dir = ".cache"
    cache_file = os.path.join(cache_dir, "tokenized_dataset")
    
    if os.path.exists(cache_file):
        try:
            return DatasetDict.load_from_disk(cache_file)
        except Exception as e:
            print(f"Warning: Failed to load cached dataset: {e}")
            # Continue with tokenization if loading fails
    
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=100  # Explicitly set max length
        )
    
    # Get columns to remove (all except 'label')
    columns_to_remove = [col for col in dataset["train"].column_names if col != 'label']
    
    # Tokenize the dataset
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),  # Parallel processing
        remove_columns=columns_to_remove  # Remove all columns except label
    )
    
    # Set the format for PyTorch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
    
    try:
        # Cache the tokenized dataset
        dataset.save_to_disk(cache_file)
    except Exception as e:
        print(f"Warning: Failed to cache dataset: {e}")
    
    return dataset

def model_init():
    """Initialize the BERT model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        gradient_checkpointing=True  # Enable gradient checkpointing
    ).to(device)
    
    return model

def get_training_args(output_dir, batch_size, num_train_epochs, learning_rate, weight_decay, warmup_ratio, gradient_accumulation_steps):
    """Get training arguments based on device type."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Base arguments that work for all devices
    args = {
        "output_dir": output_dir,
        "evaluation_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 100,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size * 2,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "logging_dir": "./logs",
        "logging_steps": 10,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "gradient_checkpointing": True,
        "max_grad_norm": 1.0,
        "dataloader_num_workers": os.cpu_count(),
        "dataloader_pin_memory": True,
    }
    
    # Add fp16 only for CUDA devices
    if device.type == "cuda":
        args["fp16"] = True
    
    return TrainingArguments(**args)

def create_subset_for_hyperopt(dataset, train_size=0.2, test_size=0.2, seed=42):
    """Create a smaller subset of the dataset for hyperparameter optimization."""
    # Sample from training set
    train_subset = dataset["train"].shuffle(seed=seed).select(range(int(len(dataset["train"]) * train_size)))
    
    # Sample from test set
    test_subset = dataset["test"].shuffle(seed=seed).select(range(int(len(dataset["test"]) * test_size)))
    
    return DatasetDict({
        "train": train_subset,
        "test": test_subset
    })

def objective(trial, dataset, model_init):
    """Objective function for hyperparameter optimization."""
    # Create a smaller subset for hyperparameter search
    subset = create_subset_for_hyperopt(dataset, train_size=0.2, test_size=0.2)
    print(f"\nUsing subset for hyperparameter optimization - Train size: {len(subset['train'])}, Test size: {len(subset['test'])}")
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 8)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4, 8])

    training_args = get_training_args(
        output_dir="./results",
        batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=subset["train"],
        eval_dataset=subset["test"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results["eval_loss"]

def train_final_model(dataset, best_params, model_init):
    """Train the final model with the best hyperparameters."""
    final_training_args = get_training_args(
        output_dir="./final_results",
        batch_size=best_params["batch_size"],
        num_train_epochs=best_params["num_train_epochs"],
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        warmup_ratio=best_params["warmup_ratio"],
        gradient_accumulation_steps=best_params["gradient_accumulation_steps"]
    )

    trainer = Trainer(
        model_init=model_init,
        args=final_training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    return trainer

def generate_classification_report(trainer, dataset, save_dir="./plots"):
    """Generate and plot classification metrics."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get predictions
    predictions = trainer.predict(dataset["test"])
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    
    # Generate classification report
    report = classification_report(labels, preds, target_names=['Negative', 'Positive'], digits=4)
    
    # Save classification report to file
    report_path = os.path.join(save_dir, f'classification_report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("====================\n\n")
        f.write(report)
    
    # Create confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # Save confusion matrix plot
    cm_plot_path = os.path.join(save_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(cm_plot_path)
    plt.close()
    
    # Calculate additional metrics
    accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'report_path': report_path,
        'cm_plot_path': cm_plot_path
    }
    
    return metrics

def plot_training_metrics(trainer, save_dir="./plots"):
    """Plot training and evaluation metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get training logs
    logs = pd.DataFrame(trainer.state.log_history)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(logs[logs['loss'].notna()]['step'], logs[logs['loss'].notna()]['loss'], label='Training Loss')
    if 'eval_loss' in logs.columns:
        eval_steps = logs[logs['eval_loss'].notna()]['step']
        eval_loss = logs[logs['eval_loss'].notna()]['eval_loss']
        plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='o')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the loss plot
    loss_plot_path = os.path.join(save_dir, f'training_loss_{timestamp}.png')
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Plot learning rate if available
    if 'learning_rate' in logs.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(logs[logs['learning_rate'].notna()]['step'], 
                logs[logs['learning_rate'].notna()]['learning_rate'])
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the learning rate plot
        lr_plot_path = os.path.join(save_dir, f'learning_rate_{timestamp}.png')
        plt.savefig(lr_plot_path)
        plt.close()
    
    return loss_plot_path

def plot_optuna_visualization(study, save_dir="./plots"):
    """Create and save Optuna visualization plots."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    history_plot_path = os.path.join(save_dir, f'optuna_history_{timestamp}.png')
    plt.savefig(history_plot_path)
    plt.close()
    
    # Plot parameter importances
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    importance_plot_path = os.path.join(save_dir, f'param_importance_{timestamp}.png')
    plt.savefig(importance_plot_path)
    plt.close()
    
    return history_plot_path, importance_plot_path

def main():
    print("Using device:", "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load and preprocess data
    data = load_and_preprocess_data("datasets/amazon_reviews.csv")
    
    # Prepare dataset
    dataset = prepare_dataset(data)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize dataset
    dataset = tokenize_dataset(dataset, tokenizer)
    
    # Print original dataset sizes
    print(f"\nFull dataset sizes:")
    print(f"Train: {len(dataset['train'])} examples")
    print(f"Test: {len(dataset['test'])} examples")
    
    # Hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, dataset, model_init),
        n_trials=3,
        timeout=3600 * 8  # 8 hour timeout
    )
    
    print("\nBest hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)
    
    # Plot Optuna results
    history_plot, importance_plot = plot_optuna_visualization(study)
    print(f"Optimization history plot saved to: {history_plot}")
    print(f"Parameter importance plot saved to: {importance_plot}")
    
    print("\nTraining final model on full dataset...")
    # Train final model on full dataset
    trainer = train_final_model(dataset, study.best_params, model_init)
    
    # Plot training metrics
    loss_plot = plot_training_metrics(trainer)
    print(f"Training metrics plot saved to: {loss_plot}")
    
    # Generate and save classification metrics
    metrics = generate_classification_report(trainer, dataset)
    print("\nClassification Report:")
    print("=====================")
    print(metrics['classification_report'])
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"\nDetailed classification report saved to: {metrics['report_path']}")
    print(f"Confusion matrix plot saved to: {metrics['cm_plot_path']}")
    
    # Save the model and tokenizer
    trainer.model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")
    
    # Final evaluation
    final_metrics = trainer.evaluate()
    print("\nFinal evaluation metrics:", final_metrics)
    
    print("Model training completed and saved!")

if __name__ == "__main__":
    main()
