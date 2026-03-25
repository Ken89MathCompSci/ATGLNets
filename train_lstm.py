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
from models import LSTMModel
from utils import calculate_nilm_metrics, save_model, load_model

def train_lstm_model(data_dict, model_params, train_params, save_dir='models'):
    """
    Train LSTM model for NILM
    
    Args:
        data_dict: Dictionary containing data loaders and metadata
        model_params: Dictionary of model parameters
        train_params: Dictionary of training parameters
        save_dir: Directory to save the model
        
    Returns:
        Trained model and training history
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
    num_layers = model_params.get('num_layers', 2)
    output_size = model_params.get('output_size', 1)
    bidirectional = model_params.get('bidirectional', False)
    
    # Create model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, bidirectional)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Extract training parameters
    lr = train_params.get('lr', 0.001)
    epochs = train_params.get('epochs', 50)
    patience = train_params.get('patience', 10)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    counter = 0
    best_model_path = None
    
    # Start training
    print(f"Starting LSTM training on {device}...")
    
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
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(save_dir, f"lstm_model_best.pth")
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
    final_model_path = os.path.join(save_dir, f"lstm_model_final.pth")
    save_model(model, model_params, train_params, metrics, final_model_path)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    val_mae = [metrics['mae'] for metrics in history['val_metrics']]
    plt.plot(val_mae, label='Validation MAE')
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"lstm_training_history.png"))
    plt.close()
    
    # Save training history to JSON
    with open(os.path.join(save_dir, 'lstm_history.json'), 'w') as f:
        json_history = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_metrics': [
                {k: float(v) for k, v in metrics.items()}
                for metrics in history['val_metrics']
            ]
        }
        json.dump(json_history, f, indent=4)
    
    return model, history, best_model_path

def train_lstm_all_appliances(house_number=1, window_size=100, save_dir='models/lstm'):
    """
    Train LSTM model on all appliances in the specified house
    
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
    print(f"Training LSTM models for {len(appliances)} appliances in house {house_number}:")
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
    
    # Train LSTM model for each appliance
    for appliance_idx, appliance_name in appliances.items():
        print(f"\n{'-'*50}")
        print(f"Training LSTM model for {appliance_name} (index {appliance_idx})")
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
                'input_size': 1,
                'hidden_size': 128,
                'num_layers': 2,
                'output_size': 1,
                'bidirectional': True
            }
            
            # Training parameters
            train_params = {
                'lr': 0.001,
                'epochs': 50,
                'patience': 10
            }
            
            # Train the model
            model, history, best_model_path = train_lstm_model(
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
            print(f"Successfully trained LSTM model for {appliance_name}")
            
        except Exception as e:
            print(f"Error training LSTM model for {appliance_name}: {str(e)}")
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
    
    print("\nLSTM training completed for all appliances!")
    print(f"Results saved to {base_save_dir}")
    
    return results, base_save_dir

if __name__ == "__main__":
    # Train LSTM models on all appliances in house 1
    results, save_dir = train_lstm_all_appliances(house_number=1)