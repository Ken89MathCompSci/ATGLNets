import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json

# Import from our modules
from data_loader import load_and_preprocess_ukdale, explore_available_appliances
from models import SimpleTransformerModel
from utils import calculate_nilm_metrics, save_model, load_model

def train_transformer_model(data_dict, model_params, train_params, save_dir='models'):
    """
    Train SimpleTransformer model for NILM
    
    Args:
        data_dict: Dictionary containing data loaders and metadata
        model_params: Dictionary of model parameters
        train_params: Dictionary of training parameters
        save_dir: Directory to save the model
        
    Returns:
        Trained model, training history, and best model path
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Extract data loaders
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    
    # Extract model parameters
    input_size = model_params.get('input_size', 1)
    hidden_size = model_params.get('hidden_size', 128)
    output_size = model_params.get('output_size', 1)
    num_layers = model_params.get('num_layers', 3)
    num_heads = model_params.get('num_heads', 4)
    dropout = model_params.get('dropout', 0.1)
    
    # Create model
    model = SimpleTransformerModel(
        input_size, 
        hidden_size, 
        output_size, 
        num_layers, 
        num_heads, 
        dropout
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Extract training parameters
    lr = train_params.get('lr', 0.0005)  # Slightly lower LR for transformer
    epochs = train_params.get('epochs', 50)
    patience = train_params.get('patience', 10)
    warmup_steps = train_params.get('warmup_steps', 1000)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.01
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        # Linear warmup followed by cosine decay
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            decay_steps = epochs * len(train_loader) - warmup_steps
            step = min(step - warmup_steps, decay_steps)
            return 0.5 * (1 + np.cos(step * np.pi / decay_steps))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': [],
        'learning_rate': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    counter = 0
    best_model_path = None
    
    # Step counter for LR scheduler
    step = 0
    
    # Start training
    print(f"Starting Transformer training on {device}...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            step += 1
            
            # Update statistics
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Update statistics
                val_loss += loss.item()
                
                # Store targets and outputs for metrics calculation
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # Calculate NILM metrics
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        metrics = calculate_nilm_metrics(all_targets, all_outputs)
        history['val_metrics'].append(metrics)
        
        # Print epoch stats
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val MAE: {metrics['mae']:.2f}, Val F1: {metrics['f1']:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            
            # Save best model
            best_model_path = os.path.join(save_dir, f"transformer_model_best.pth")
            save_model(model, model_params, train_params, metrics, best_model_path)
            print(f"Model saved to {best_model_path}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            
            if counter >= patience:
                print("Early stopping triggered")
                break
    
    print("Training completed!")
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"transformer_model_final.pth")
    save_model(model, model_params, train_params, metrics, final_model_path)
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(2, 2, 2)
    val_mae = [metrics['mae'] for metrics in history['val_metrics']]
    plt.plot(val_mae, label='Validation MAE')
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # Plot F1 Score
    plt.subplot(2, 2, 3)
    val_f1 = [metrics['f1'] for metrics in history['val_metrics']]
    plt.plot(val_f1, label='Validation F1 Score')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Plot Learning Rate
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"transformer_training_history.png"))
    plt.close()
    
    # Save training history to JSON
    with open(os.path.join(save_dir, 'transformer_history.json'), 'w') as f:
        json_history = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'learning_rate': [float(x) for x in history['learning_rate']],
            'val_metrics': [
                {k: float(v) for k, v in metrics.items()}
                for metrics in history['val_metrics']
            ]
        }
        json.dump(json_history, f, indent=4)
    
    return model, history, best_model_path

def train_transformer_all_appliances(house_number=1, window_size=100, save_dir='models/transformer'):
    """
    Train SimpleTransformer model on all appliances in the specified house
    
    Args:
        house_number: House number in the UK-DALE dataset
        window_size: Window size for input sequences
        save_dir: Directory to save the models
    
    Returns:
        Dictionary mapping appliance names to their training results
    """
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join(save_dir, f"house{house_number}_{timestamp}")
    
    # Create main save directory
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Load data file
    file_path = f"preprocessed_datasets/ukdale/ukdale{house_number}.mat"
    
    # Get available appliances
    appliances = explore_available_appliances(file_path)
    print(f"Training Transformer models for {len(appliances)} appliances in house {house_number}:")
    for idx, name in appliances.items():
        print(f"  Index {idx}: {name}")
    
    # Save configuration
    config = {
        'house_number': house_number,
        'window_size': window_size,
        'timestamp': timestamp,
        'appliances': appliances
    }
    
    with open(os.path.join(base_save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Dictionary to store results
    results = {}
    
    # Train Transformer model for each appliance
    for appliance_idx, appliance_name in appliances.items():
        print(f"\n{'-'*50}")
        print(f"Training Transformer model for {appliance_name} (index {appliance_idx})")
        print(f"{'-'*50}\n")
        
        # Create appliance-specific save directory
        appliance_dir = os.path.join(base_save_dir, f"{appliance_name}")
        os.makedirs(appliance_dir, exist_ok=True)
        
        try:
            # Load data for this appliance
            data_dict = load_and_preprocess_ukdale(
                file_path,
                appliance_idx,
                window_size=window_size,
                target_size=1
            )
            
            # Model parameters
            model_params = {
                'input_size': 1,              # Input size (power)
                'hidden_size': 128,           # Hidden dimension
                'output_size': 1,             # Output size (prediction)
                'num_layers': 3,              # Number of transformer encoder layers
                'num_heads': 4,               # Number of attention heads
                'dropout': 0.1                # Dropout probability
            }
            
            # Training parameters
            train_params = {
                'lr': 0.0005,                 # Initial learning rate (lower for transformer)
                'epochs': 50,                 # Maximum number of epochs
                'patience': 10,               # Early stopping patience
                'warmup_steps': 1000          # Warmup steps for learning rate
            }
            
            # Train the model
            model, history, best_model_path = train_transformer_model(
                data_dict, 
                model_params, 
                train_params, 
                save_dir=appliance_dir
            )
            
            # Store results
            results[appliance_name] = {
                'model_path': best_model_path,
                'appliance_index': appliance_idx,
                'final_metrics': history['val_metrics'][-1] if history['val_metrics'] else None
            }
            
            # Log success
            print(f"Successfully trained Transformer model for {appliance_name}")
            
        except Exception as e:
            print(f"Error training Transformer model for {appliance_name}: {str(e)}")
            # Continue with the next appliance
    
    # Save summary of results
    summary = {
        'timestamp': timestamp,
        'house_number': house_number,
        'results': {
            name: {
                'model_path': info['model_path'],
                'final_metrics': {k: float(v) for k, v in info['final_metrics'].items()} if info['final_metrics'] else None
            }
            for name, info in results.items()
        }
    }
    
    with open(os.path.join(base_save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\nTransformer training completed for all appliances!")
    print(f"Results saved to {base_save_dir}")
    
    return results, base_save_dir

if __name__ == "__main__":
    # Train Transformer models on all appliances in house 1
    results, save_dir = train_transformer_all_appliances(house_number=1)