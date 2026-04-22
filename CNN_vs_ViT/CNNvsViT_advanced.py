import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

import timm
from transformers import ViTForImageClassification

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import gc
import os
import time
import multiprocessing
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler  # [New] 혼합 정밀도(AMP)

# -------------------------
# 결과 저장 디렉터리
# -------------------------
# 폴더명을 'improved_results'로 설정
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
BASE_RESULT_DIR = os.path.join("results", f"improved_{RUN_TIMESTAMP}")
CM_DIR = os.path.join(BASE_RESULT_DIR, "confusion_matrix")
LOSS_DIR = os.path.join(BASE_RESULT_DIR, "loss_curves")

os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)

# -------------------------
# 로그 설정
# -------------------------
LOG_PATH = os.path.join(BASE_RESULT_DIR, "experiment_log.txt")
log_file = open(LOG_PATH, "w", buffering=1)


def log_print(message=""):
    print(message)
    log_file.write(str(message) + "\n")


# -------------------------
# [개선됨] 하이퍼파라미터
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# [New] AMP 사용 시 메모리 절약되므로 배치 사이즈 증가 가능 (속도 향상)
BATCH_SIZE = 32
EPOCHS = 5  # [New] Scheduler 효과를 보기 위해 에폭 소폭 증가 (또는 3 유지 가능)
LR = 3e-4

# [New] CPU 코어 수에 맞춰 워커 설정 (리눅스 필수)
NUM_WORKERS = min(4, multiprocessing.cpu_count())

MODEL_LIST = {
    "ResNet18": "resnet18",
    "EfficientNet-B0": "efficientnet_b0",
    "ConvNeXt-Tiny": "convnext_tiny",
    "ViT-B16": "ViT-B16"
}

# 설정 기록
with open(os.path.join(BASE_RESULT_DIR, "config.txt"), "w") as f:
    f.write(f"NOTE: This is the OPTIMIZED (Improved) code.\n")
    f.write(f"Techniques: ImageNet Norm, Cosine Scheduler, AMP, Label Smoothing, Num_workers\n")
    f.write(f"DEVICE: {DEVICE}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"EPOCHS: {EPOCHS}\n")
    f.write(f"NUM_WORKERS: {NUM_WORKERS}\n")


# -------------------------
# [개선됨] DataLoader
# -------------------------
def get_dataloaders(dataset):
    # [New] ImageNet 통계값 사용 (Pre-trained 모델 성능 극대화)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)  # [New] 최적화된 정규화
    ])

    if dataset == "cifar10":
        train_set = CIFAR10("./data", train=True, download=True, transform=transform)
        test_set = CIFAR10("./data", train=False, download=True, transform=transform)
        num_classes = 10
    else:
        train_set = CIFAR100("./data", train=True, download=True, transform=transform)
        test_set = CIFAR100("./data", train=False, download=True, transform=transform)
        num_classes = 100

    # [New] num_workers, pin_memory 적용 (데이터 로딩 속도 향상)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    return train_loader, test_loader, num_classes, train_set.classes


# -------------------------
# Model Loader
# -------------------------
def load_model(model_key, num_classes):
    if model_key == "ViT-B16":
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    else:
        model = timm.create_model(
            model_key,
            pretrained=True,
            num_classes=num_classes
        )
    return model


# -------------------------
# [개선됨] Train Function (AMP + Scheduler)
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        # [New] AMP (Mixed Precision) 적용: 전방 연산
        with autocast():
            outputs = model(x)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, y)

        # [New] AMP 적용: 역전파 및 가중치 업데이트
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate_predictions(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            # Eval 때는 AMP 선택사항이나, 속도를 위해 켤 수 있음
            with autocast():
                outputs = model(x)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                preds = torch.argmax(logits, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return np.array(y_true), np.array(y_pred)


# -------------------------
# Main Experiment Loop
# -------------------------
summary_results = []
classwise_results = []
time_results = []

total_start_time = time.time()

for dataset in ["cifar10", "cifar100"]:
    log_print(f"\n===== DATASET: {dataset.upper()} =====")
    train_loader, test_loader, num_classes, class_names = get_dataloaders(dataset)

    dataset_loss_dict = {}

    for model_label, model_key in MODEL_LIST.items():
        log_print(f"\nTraining {model_label} (Optimized)...")

        model_start_time = time.time()

        model = load_model(model_key, num_classes).to(DEVICE)

        # [New] Optimizer & Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

        # [New] Cosine Annealing Scheduler: LR을 부드럽게 감소시킴
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # [New] Label Smoothing: 과적합 방지
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # [New] GradScaler for AMP
        scaler = GradScaler()

        loss_history = []
        for epoch in range(EPOCHS):
            avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
            scheduler.step()  # 에폭마다 LR 업데이트
            loss_history.append(avg_loss)

            # 현재 LR 출력 (로그 확인용)
            current_lr = optimizer.param_groups[0]['lr']
            log_print(f"  Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        dataset_loss_dict[model_label] = loss_history

        # Evaluation
        y_true, y_pred = evaluate_predictions(model, test_loader)

        model_end_time = time.time()
        elapsed_time = model_end_time - model_start_time
        log_print(f"  >> Time taken: {elapsed_time:.2f} sec")

        acc = accuracy_score(y_true, y_pred)
        summary_results.append({
            "Dataset": dataset.upper(),
            "Model": model_label,
            "Accuracy (%)": round(acc * 100, 2)
        })

        time_results.append({
            "Dataset": dataset.upper(),
            "Model": model_label,
            "Training_Time_Sec": round(elapsed_time, 2)
        })

        if dataset == "cifar10":
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, cmap="Greens")  # 색상을 바꿔서 Baseline과 구분
            plt.title(f"[Optimized] {dataset.upper()} - {model_label}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(CM_DIR, f"{dataset.upper()}_{model_label}.png"), dpi=150)
            plt.close()

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        for i, class_name in enumerate(class_names):
            classwise_results.append({
                "Dataset": dataset.upper(),
                "Model": model_label,
                "Class": class_name,
                "Precision": round(precision[i], 4),
                "Recall": round(recall[i], 4),
                "F1-score": round(f1[i], 4),
                "Support": int(support[i])
            })

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # [New] Loss Curve 저장
    plt.figure(figsize=(10, 6))
    for m_label, losses in dataset_loss_dict.items():
        plt.plot(range(1, EPOCHS + 1), losses, marker='o', label=m_label)
    plt.title(f"Training Loss (Optimized) - {dataset.upper()}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(LOSS_DIR, f"{dataset.upper()}_loss_curve.png"), dpi=150)
    plt.close()

total_end_time = time.time()
total_duration = total_end_time - total_start_time
log_print(f"\nTotal Improved Experiment Duration: {total_duration:.2f} seconds")

# -------------------------
# Save Results
# -------------------------
summary_df = pd.DataFrame(summary_results)
classwise_df = pd.DataFrame(classwise_results)
time_df = pd.DataFrame(time_results)

summary_df.to_csv(os.path.join(BASE_RESULT_DIR, "new_summary.csv"), index=False)
classwise_df.to_csv(os.path.join(BASE_RESULT_DIR, "new_classwise.csv"), index=False)
time_df.to_csv(os.path.join(BASE_RESULT_DIR, "new_execution_times.csv"), index=False)

log_print("\n=== [Optimized] Execution Time Summary ===")
log_print(time_df.to_string(index=False))

log_print("\n=== [Optimized] Accuracy Summary ===")
log_print(summary_df.to_string(index=False))

log_file.close()