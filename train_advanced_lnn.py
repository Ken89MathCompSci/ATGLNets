import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json

from data_loader import explore_available_appliances, load_house, H5_PATH, SPLIT_RANGES
from models import AdvancedLiquidNetworkModel
from utils import calculate_nilm_metrics, save_model


def get_threshold_for_appliance(appliance_name):
    """Physical watt threshold for on/off detection (applied after inverse-transform)."""
    return 0.5 if appliance_name == 'washer_dryer' else 10.0


def train_advanced_lnn_model(data_dict, model_params, train_params, save_dir='models'):
    """
    Train Advanced Liquid Neural Network model for NILM.

    Returns:
        Trained model, training history, test metrics, and best model path
    """
    os.makedirs(save_dir, exist_ok=True)

    train_loader       = data_dict['train_loader']
    val_loader         = data_dict['val_loader']
    test_loader        = data_dict['test_loader']
    appliance_scaler   = data_dict['appliance_scaler']
    appliance_name     = data_dict.get('appliance_name', 'unknown')
    threshold          = get_threshold_for_appliance(appliance_name)

    input_size  = model_params.get('input_size', 1)
    hidden_size = model_params.get('hidden_size', 64)
    output_size = model_params.get('output_size', 1)
    num_layers  = model_params.get('num_layers', 2)
    dt          = model_params.get('dt', 0.1)

    model  = AdvancedLiquidNetworkModel(input_size, hidden_size, output_size, num_layers, dt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)

    lr       = train_params.get('lr', 0.001)
    epochs   = train_params.get('epochs', 80)
    patience = train_params.get('patience', 20)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history         = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss   = float('inf')
    counter         = 0
    best_model_path = None

    print(f"Starting Advanced LNN training for '{appliance_name}' on {device}...")

    def inverse(arr):
        if appliance_scaler is None:
            return arr
        return appliance_scaler.inverse_transform(
            arr.reshape(-1, 1)).flatten()

    for epoch in range(epochs):
        # --- Training ---
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

        # --- Validation ---
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
        scheduler.step(avg_val_loss)

        # Inverse-transform → physical watts before metrics
        t_phys = inverse(np.concatenate(all_targets))
        o_phys = inverse(np.concatenate(all_outputs))
        metrics = calculate_nilm_metrics(t_phys, o_phys, threshold=threshold)
        history['val_metrics'].append(metrics)

        print(f"Epoch {epoch+1}/{epochs}  Train: {avg_train_loss:.6f}  "
              f"Val: {avg_val_loss:.6f}  MAE: {metrics['mae']:.2f}W  "
              f"SAE: {metrics['sae']:.4f}  F1: {metrics['f1']:.4f}")

        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss   = avg_val_loss
            counter         = 0
            best_model_path = os.path.join(
                save_dir,
                f"advanced_lnn_{appliance_name.replace(' ', '_')}_best.pth")
            save_model(model, model_params, train_params, metrics, best_model_path)
            print(f"  -> Best model saved to {best_model_path}")
        else:
            counter += 1
            print(f"  EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training completed!")

    # --- Test evaluation ---
    print("Evaluating on test set...")
    model.eval()
    test_loss        = 0.0
    all_test_targets = []
    all_test_outputs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            all_test_targets.append(targets.cpu().numpy())
            all_test_outputs.append(outputs.cpu().numpy())

    avg_test_loss  = test_loss / len(test_loader)
    tt_phys        = inverse(np.concatenate(all_test_targets))
    to_phys        = inverse(np.concatenate(all_test_outputs))
    test_metrics   = calculate_nilm_metrics(tt_phys, to_phys, threshold=threshold)

    print(f"Test Loss: {avg_test_loss:.6f}  Test Metrics: {test_metrics}")

    # --- Aggregates (mean/var over epochs) ---
    def _agg(key):
        vals = [m[key] for m in history['val_metrics']]
        return float(np.mean(vals)), float(np.var(vals))

    mae_mean,  mae_var  = _agg('mae')
    sae_mean,  sae_var  = _agg('sae')
    f1_mean,   f1_var   = _agg('f1')
    prec_mean, prec_var = _agg('precision')
    rec_mean,  rec_var  = _agg('recall')

    aggregates = {
        'train_loss_mean':    float(np.mean(history['train_loss'])),
        'train_loss_var':     float(np.var(history['train_loss'])),
        'val_loss_mean':      float(np.mean(history['val_loss'])),
        'val_loss_var':       float(np.var(history['val_loss'])),
        'val_mae_mean':       mae_mean,  'val_mae_var':       mae_var,
        'val_sae_mean':       sae_mean,  'val_sae_var':       sae_var,
        'val_f1_mean':        f1_mean,   'val_f1_var':        f1_var,
        'val_precision_mean': prec_mean, 'val_precision_var': prec_var,
        'val_recall_mean':    rec_mean,  'val_recall_var':    rec_var,
        'test_mae':       float(test_metrics['mae']),
        'test_sae':       float(test_metrics['sae']),
        'test_f1':        float(test_metrics['f1']),
        'test_precision': float(test_metrics['precision']),
        'test_recall':    float(test_metrics['recall']),
        'test_loss':      float(avg_test_loss),
    }
    print("Aggregates:"); print(json.dumps(aggregates, indent=2))

    # --- 4-panel plot ---
    val_mae = [m['mae'] for m in history['val_metrics']]
    val_sae = [m['sae'] for m in history['val_metrics']]
    val_f1  = [m['f1']  for m in history['val_metrics']]

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'],   label='Val Loss',   color='red')
    plt.title(f'Loss – {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(val_mae, label='Val MAE', color='red')
    plt.axhline(test_metrics['mae'], label='Test MAE', color='green', linestyle='--')
    plt.title(f'MAE – {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MAE (W)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(val_sae, label='Val SAE', color='red')
    plt.axhline(test_metrics['sae'], label='Test SAE', color='green', linestyle='--')
    plt.title(f'SAE – {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('SAE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(val_f1, label='Val F1', color='red')
    plt.axhline(test_metrics['f1'], label='Test F1', color='green', linestyle='--')
    plt.title(f'F1 Score – {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('F1')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(
        save_dir,
        f"advanced_lnn_{appliance_name.replace(' ', '_')}_metrics.png"),
        dpi=300, bbox_inches='tight')
    plt.close()

    # --- Per-appliance JSON ---
    config = {
        'appliance':   appliance_name,
        'dataset':     'UKDALE',
        'window_size': data_dict.get('window_size', 100),
        'dataset_splits': {k: {'start': v[0], 'end': v[1]} for k, v in SPLIT_RANGES.items()},
        'model_params': model_params,
        'train_params': train_params,
        'final_metrics': {
            'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'val_loss':   history['val_loss'][-1]   if history['val_loss']   else None,
            'test_loss':  float(avg_test_loss),
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'aggregates':   aggregates,
        }
    }
    with open(os.path.join(
            save_dir,
            f"advanced_lnn_{appliance_name.replace(' ', '_')}_history.json"),
            'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics, best_model_path


def train_advanced_lnn_all_appliances(house_number=1, window_size=100,
                                      save_dir='models/advanced_lnn'):
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

    json.dump(
        {'house_number': house_number, 'window_size': window_size,
         'timestamp': timestamp, 'model': 'advanced_lnn',
         'dataset_splits': {k: {'start': v[0], 'end': v[1]}
                            for k, v in SPLIT_RANGES.items()}},
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

            model, history, test_metrics, best_path = train_advanced_lnn_model(
                data_dict,
                model_params={'input_size': 1, 'hidden_size': 64,
                              'output_size': 1, 'num_layers': 2, 'dt': 0.1},
                train_params={'lr': 0.001, 'epochs': 80, 'patience': 20},
                save_dir=appliance_dir
            )

            results[appliance_name] = {
                'model_path':    best_path,
                'final_metrics': {k: float(v) for k, v in test_metrics.items()},
            }
            print(f"Done: {appliance_name}")

        except Exception as e:
            import traceback
            print(f"Error on {appliance_name}: {e}")
            traceback.print_exc()

    summary = {
        'timestamp':    timestamp,
        'house_number': house_number,
        'model':        'advanced_lnn',
        'dataset':      'UKDALE',
        'dataset_splits': {k: {'start': v[0], 'end': v[1]}
                           for k, v in SPLIT_RANGES.items()},
        'results': results,
    }
    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nAdvanced LNN training complete. Results saved to {base_save_dir}")
    return results, base_save_dir


if __name__ == "__main__":
    results, save_dir = train_advanced_lnn_all_appliances(house_number=1)
