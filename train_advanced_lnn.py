import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json

from data_loader import explore_available_appliances, load_house, H5_PATH
from models import AdvancedLiquidNetworkModel
from utils import calculate_nilm_metrics, save_model


def train_advanced_lnn_model(data_dict, model_params, train_params, save_dir='models'):
    """
    Train Advanced Liquid Neural Network model for NILM.

    Returns:
        Trained model, training history, and best model path
    """
    os.makedirs(save_dir, exist_ok=True)

    train_loader = data_dict['train_loader']
    val_loader   = data_dict['val_loader']

    input_size  = model_params.get('input_size', 1)
    hidden_size = model_params.get('hidden_size', 128)
    output_size = model_params.get('output_size', 1)
    num_layers  = model_params.get('num_layers', 2)
    dt          = model_params.get('dt', 0.1)

    model  = AdvancedLiquidNetworkModel(input_size, hidden_size, output_size, num_layers, dt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)

    lr       = train_params.get('lr', 0.001)
    epochs   = train_params.get('epochs', 50)
    patience = train_params.get('patience', 10)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss   = float('inf')
    counter         = 0
    best_model_path = None

    print(f"Starting Advanced LNN training on {device}...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss   = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss    = 0.0
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs  = model(inputs)
                val_loss += criterion(outputs, targets).item()
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        metrics = calculate_nilm_metrics(all_targets, all_outputs)
        history['val_metrics'].append(metrics)

        print(f"Epoch {epoch+1}/{epochs}  Train Loss: {avg_train_loss:.6f}  "
              f"Val Loss: {avg_val_loss:.6f}  MAE: {metrics['mae']:.4f}  F1: {metrics['f1']:.4f}")

        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss   = avg_val_loss
            counter         = 0
            best_model_path = os.path.join(save_dir, "advanced_lnn_model_best.pth")
            save_model(model, model_params, train_params, metrics, best_model_path)
            print(f"  -> Best model saved to {best_model_path}")
        else:
            counter += 1
            print(f"  EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training completed!")
    final_path = os.path.join(save_dir, "advanced_lnn_model_final.pth")
    save_model(model, model_params, train_params, metrics, final_path)

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'],   label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([m['mae'] for m in history['val_metrics']], label='Val MAE')
    plt.plot([m['f1']  for m in history['val_metrics']], label='Val F1')
    plt.title('Metrics')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "advanced_lnn_training_history.png"))
    plt.close()

    with open(os.path.join(save_dir, 'advanced_lnn_history.json'), 'w') as f:
        json.dump({
            'train_loss':  [float(x) for x in history['train_loss']],
            'val_loss':    [float(x) for x in history['val_loss']],
            'val_metrics': [{k: float(v) for k, v in m.items()} for m in history['val_metrics']]
        }, f, indent=4)

    return model, history, best_model_path


def train_advanced_lnn_all_appliances(house_number=1, window_size=100, save_dir='models/advanced_lnn'):
    """
    Train Advanced LNN on all target appliances for the specified house.
    """
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join(save_dir, f"house{house_number}_{timestamp}")
    os.makedirs(base_save_dir, exist_ok=True)

    file_path  = f"preprocessed_datasets/ukdale/ukdale{house_number}.mat"
    appliances = explore_available_appliances(file_path)

    print(f"Training Advanced LNN for {len(appliances)} appliances in house {house_number}:")
    for idx, name in appliances.items():
        print(f"  {idx}: {name}")

    print("\nPre-loading house data...")
    house_data = load_house(H5_PATH, house_number, window_size=window_size)

    json.dump({'house_number': house_number, 'window_size': window_size,
               'timestamp': timestamp, 'model': 'advanced_lnn'},
              open(os.path.join(base_save_dir, 'config.json'), 'w'), indent=4)

    results = {}
    for appliance_idx, appliance_name in appliances.items():
        print(f"\n{'-'*50}")
        print(f"Training Advanced LNN for {appliance_name}")
        print(f"{'-'*50}")

        appliance_dir = os.path.join(base_save_dir, appliance_name)
        os.makedirs(appliance_dir, exist_ok=True)

        try:
            data_dict = house_data.get(appliance_name)
            if data_dict is None:
                print(f"  [SKIP] '{appliance_name}' not found")
                continue

            model, history, best_path = train_advanced_lnn_model(
                data_dict,
                model_params={'input_size': 1, 'hidden_size': 128,
                              'output_size': 1, 'num_layers': 2, 'dt': 0.1},
                train_params={'lr': 0.001, 'epochs': 50, 'patience': 10},
                save_dir=appliance_dir
            )

            results[appliance_name] = {
                'model_path':    best_path,
                'final_metrics': history['val_metrics'][-1] if history['val_metrics'] else None
            }
            print(f"Done: {appliance_name}")

        except Exception as e:
            print(f"Error on {appliance_name}: {e}")

    summary = {
        'timestamp': timestamp, 'house_number': house_number, 'model': 'advanced_lnn',
        'results': {
            name: {'model_path': info['model_path'],
                   'final_metrics': {k: float(v) for k, v in info['final_metrics'].items()}
                   if info['final_metrics'] else None}
            for name, info in results.items()
        }
    }
    with open(os.path.join(base_save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nAdvanced LNN training complete. Results saved to {base_save_dir}")
    return results, base_save_dir


if __name__ == "__main__":
    results, save_dir = train_advanced_lnn_all_appliances(house_number=1)
