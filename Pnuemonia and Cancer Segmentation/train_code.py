import os
from torch import nn
import torch.optim as optim
import torch
import time
import pandas as pd
from metrics import dice_score, precision, recall, specificity, f1_score, rmse, binary_iou, binary_dice, binary_f1
import numpy as np
from tqdm.auto import tqdm


def cancer_seg_train_model(model, train_loader, val_loader, device, name, num_epochs=50):
    save_dir = name
    os.makedirs(save_dir, exist_ok=True)

    # Losses
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    
    # Log DataFrame with custom column order
    columns = [
        'Epoch', 'Total Loss', 'Dice Score', 'Time (s)',
        'Precision', 'Recall', 'F1 Score', 'Specificity', 
        'Val Total Loss',  'Val Dice Score',  'Val Time (s)', 
        'Val Precision', 'Val Recall', 'Val F1 Score', 'Val Specificity',

    ]
    log_df = pd.DataFrame(columns=columns)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        metrics = {col: 0.0 for col in columns}  # Initialize metrics dict

        for phase in ['train', 'val']:
            dataloader = train_loader if phase == 'train' else val_loader
            model.train() if phase == 'train' else model.eval()

            batch_metrics = {
                'total_loss': [],
                'dice': [], 'precision': [], 'recall': [], 'specificity': [],
                
            }

            start_time = time.time()

            with torch.set_grad_enabled(phase == 'train'):
                for noisy_img, mask in tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch+1}", leave=False):
                    noisy_img = noisy_img.to(device)
                    mask = mask.to(device)

                    seg_mask_logits = model(noisy_img)

                    # Losses
                    loss_seg = bce_loss(seg_mask_logits, mask)
                    total_loss = loss_seg

                    if phase == 'train':
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                    # Segmentation metrics
                    seg_probs = torch.sigmoid(seg_mask_logits)
                    dice = dice_score(seg_probs, mask).item()
                    prec = precision(seg_probs, mask).item()
                    rec = recall(seg_probs, mask).item()
                    spec = specificity(seg_probs, mask).item()
                    f1 = f1_score(prec, rec)

                    # Append batch metrics
                    batch_metrics['total_loss'].append(total_loss.item())
                    batch_metrics['dice'].append(dice)
                    batch_metrics['precision'].append(prec)
                    batch_metrics['recall'].append(rec)
                    batch_metrics['specificity'].append(spec)
                  

            # Aggregate metrics
            end_time = time.time()
            avg = {k: np.mean(v) for k, v in batch_metrics.items()}
            time_taken = end_time - start_time

            # Update metrics dict
            prefix = '' if phase == 'train' else 'Val '
            metrics[f'{prefix}Total Loss'] = avg['total_loss']
            metrics[f'{prefix}Dice Score'] = avg['dice']
            metrics[f'{prefix}Time (s)'] = time_taken
            metrics[f'{prefix}Precision'] = avg['precision']
            metrics[f'{prefix}Recall'] = avg['recall']
            metrics[f'{prefix}F1 Score'] = f1_score(avg['precision'], avg['recall'])
            metrics[f'{prefix}Specificity'] = avg['specificity']
      

        # Append to log
        log_df = pd.concat([log_df, pd.DataFrame([metrics])], ignore_index=True)

        # Save best model
        if metrics['Val Total Loss'] < best_val_loss:
            best_val_loss = metrics['Val Total Loss']
            best_model_path = os.path.join(save_dir, f'{name}_best_val_loss.pt')
            torch.save(model.state_dict(), best_model_path)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {metrics['Total Loss']:.4f} | "
              f"Train Dice: {metrics['Dice Score']:.4f} | "
              f"Val Loss: {metrics['Val Total Loss']:.4f} | "
              f"Val Dice: {metrics['Val Dice Score']:.4f} | "
              )

    # Save logs
    log_path = os.path.join(save_dir, f'{name}_Training.csv')
    log_df.to_csv(log_path, index=False, float_format='%.4f')
    print(f"Training complete. Logs saved to {log_path}")

    return model




# --- Training function ---
def covid_train_binary_segmentation(model, train_loader, val_loader, num_epochs, device, save_path):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    best_val_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        iou_scores, dice_scores, f1_scores = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).bool()

            iou_scores.append(binary_iou(preds, masks.bool()))
            dice_scores.append(binary_dice(preds, masks.bool()))
            f1_scores.append(binary_f1(preds, masks.bool()))

            pbar.set_postfix({
                'Loss': f"{train_loss / (len(iou_scores)):.4f}",
                'Dice': f"{np.nanmean(dice_scores):.4f}",
                'IoU': f"{np.nanmean(iou_scores):.4f}",
                'F1': f"{np.nanmean(f1_scores):.4f}"
            }, refresh=True)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_ious, val_dices, val_f1s = [], [], []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                masks = masks.to(device).float()

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).bool()

                val_ious.append(binary_iou(preds, masks.bool()))
                val_dices.append(binary_dice(preds, masks.bool()))
                val_f1s.append(binary_f1(preds, masks.bool()))

        avg_val_dice = np.nanmean(val_dices)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"Val IoU: {np.nanmean(val_ious):.4f} | Dice: {avg_val_dice:.4f} | F1: {np.nanmean(val_f1s):.4f}")

        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), f"{save_path}/best_model.pth")
            print("✅ Best model saved!")

    torch.save(model.state_dict(), f"{save_path}/final_model.pth")
    print("✅ Final model saved.")
