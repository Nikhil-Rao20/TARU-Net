import os
import time
import numpy as np
import pandas as pd
import torch
import cv2
from torch import nn
from tqdm.auto import tqdm
from metrics import dice_score, precision, recall, specificity, f1_score, rmse, binary_dice, binary_f1, binary_iou

def seg_test_model(model, test_loader, device, name):
    save_dir = name
    os.makedirs(save_dir, exist_ok=True)

    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    # Metrics handler

    # Column order
    columns = ['Total Loss','Dice Score', 'Time (s)',
        'Precision', 'Recall', 'F1 Score', 'Specificity', 
        
    ]

    # Initialize metrics dictionary
    metrics = {col: 0.0 for col in columns}

    # Batch metrics
    batch_metrics = {'total_loss': [],'dice': [], 'precision': [], 'recall': [], 'specificity': []}

    model.eval()  # Set model to evaluation mode
    start_time = time.time()

    with torch.no_grad():
        for noisy_img, mask in tqdm(test_loader, desc=f"TEST- Epoch ", leave=False):
            
            noisy_img = noisy_img.to(device)
            mask = mask.to(device)

            seg_mask_logits = model(noisy_img)

            # Losses
            loss_seg = bce_loss(seg_mask_logits, mask)
            total_loss = loss_seg

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
    avg = {k: np.mean(v) for k, v in batch_metrics.items()}
    time_taken = time.time() - start_time

    # Fill metrics dictionary (test = validation phase)
    metrics.update({
        'Dice Score': avg['dice'],
        'Time (s)': time_taken,
        'Precision': avg['precision'],
        'Recall': avg['recall'],
        'F1 Score': f1_score(avg['precision'], avg['recall']),
        'Specificity': avg['specificity'],
  

        
    })

    # Create DataFrame and save
    log_df = pd.DataFrame([metrics])
    log_path = os.path.join(save_dir, f'{name}_Testing.csv')
    log_df.to_csv(log_path, index=False, float_format='%.4f')

    print(f"Test complete. Results saved to {log_path}")
    print(f"Test Dice: {avg['dice']:.4f}")

    return log_df



# --- Testing Function ---
def covid_test_binary_model(model, test_loader, device='cuda', save_preds=True, save_dir="preds"):
    model.eval()
    model.to(device)

    iou_scores, dice_scores, f1_scores = [], [], []

    if save_preds:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).bool()

            iou_scores.append(binary_iou(preds, masks.bool()))
            dice_scores.append(binary_dice(preds, masks.bool()))
            f1_scores.append(binary_f1(preds, masks.bool()))

            if save_preds:
                for b in range(images.size(0)):
                    pred_np = preds[b][0].cpu().numpy().astype(np.uint8) * 255
                    mask_np = masks[b][0].cpu().numpy().astype(np.uint8) * 255
                    img_np = images[b][0].cpu().numpy() * 255

                    cv2.imwrite(os.path.join(save_dir, f"img_{idx}_{b}.png"), img_np)
                    cv2.imwrite(os.path.join(save_dir, f"gt_{idx}_{b}.png"), mask_np)
                    cv2.imwrite(os.path.join(save_dir, f"pred_{idx}_{b}.png"), pred_np)

    print("\n✅ Test Results:")
    print(f"IoU:  {np.nanmean(iou_scores):.4f}")
    print(f"Dice: {np.nanmean(dice_scores):.4f}")
    print(f"F1:   {np.nanmean(f1_scores):.4f}")
    
