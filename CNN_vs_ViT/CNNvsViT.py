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
import time  # [추가] 시간 측정을 위해 필요
from datetime import datetime

# -------------------------
# 결과 저장 디렉터리
# -------------------------
# 폴더명을 'baseline_results'로 하여 최적화 버전과 구분하기 쉽게 함
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
BASE_RESULT_DIR = os.path.join("results", f"baseline_{RUN_TIMESTAMP}")
CM_DIR = os.path.join(BASE_RESULT_DIR, "confusion_matrix")

os.makedirs(CM_DIR, exist_ok=True)

# -------------------------
# 텍스트 로그 파일 설정
# -------------------------
LOG_PATH = os.path.join(BASE_RESULT_DIR, "experiment_log.txt")
log_file = open(LOG_PATH, "w", buffering=1)


def log_print(message=""):
    print(message)
    log_file.write(str(message) + "\n")


# -------------------------
# 기본 설정 (최적화 전 상태)
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16  # 최적화 전: 작은 배치 사이즈
EPOCHS = 3
LR = 3e-4

MODEL_LIST = {
    "ResNet18": "resnet18",
    "EfficientNet-B0": "efficientnet_b0",
    "ConvNeXt-Tiny": "convnext_tiny",
    "ViT-B16": "ViT-B16"
}

# -------------------------
# 설정 기록
# -------------------------
with open(os.path.join(BASE_RESULT_DIR, "config.txt"), "w") as f:
    f.write(f"DEVICE: {DEVICE}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"EPOCHS: {EPOCHS}\n")
    f.write(f"NOTE: This is the UNOPTIMIZED baseline code.\n")


# -------------------------
# DataLoader (최적화 전: num_workers 없음)
# -------------------------
def get_dataloaders(dataset):
    # 최적화 전: 단순 정규화
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    if dataset == "cifar10":
        train_set = CIFAR10("./data", train=True, download=True, transform=transform)
        test_set = CIFAR10("./data", train=False, download=True, transform=transform)
        num_classes = 10
    else:
        train_set = CIFAR100("./data", train=True, download=True, transform=transform)
        test_set = CIFAR100("./data", train=False, download=True, transform=transform)
        num_classes = 100

    # [중요] num_workers 기본값(0) 사용 -> 데이터 로딩 병목 발생 유도
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    class_names = train_set.classes

    return train_loader, test_loader, num_classes, class_names


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
# Train / Eval
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


def evaluate_predictions(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            outputs = model(x)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())
    return np.array(y_true), np.array(y_pred)


# -------------------------
# Experiment Loop
# -------------------------
summary_results = []
classwise_results = []
time_results = []  # [추가] 시간 기록용 리스트

total_start_time = time.time()  # 전체 실험 시작 시간

for dataset in ["cifar10", "cifar100"]:

    log_print(f"\n===== DATASET: {dataset.upper()} =====")
    train_loader, test_loader, num_classes, class_names = get_dataloaders(dataset)

    for model_label, model_key in MODEL_LIST.items():

        log_print(f"\nTraining {model_label} ...")

        # [추가] 모델별 학습 시작 시간 측정
        model_start_time = time.time()

        model = load_model(model_key, num_classes).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            train_one_epoch(model, train_loader, optimizer, criterion)
            log_print(f"  Epoch {epoch + 1}/{EPOCHS} done")

        # Evaluation
        y_true, y_pred = evaluate_predictions(model, test_loader)

        # [추가] 모델별 학습 종료 시간 측정
        model_end_time = time.time()
        elapsed_time = model_end_time - model_start_time

        log_print(f"  >> Time taken for {model_label}: {elapsed_time:.2f} seconds")

        # 결과 저장
        acc = accuracy_score(y_true, y_pred)
        summary_results.append({
            "Dataset": dataset.upper(),
            "Model": model_label,
            "Accuracy (%)": round(acc * 100, 2)
        })

        # [추가] 시간 결과 저장
        time_results.append({
            "Dataset": dataset.upper(),
            "Model": model_label,
            "Training_Time_Sec": round(elapsed_time, 2)
        })

        # Confusion Matrix (CIFAR10만)
        if dataset == "cifar10":
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, cmap="Blues")
            plt.title(f"{dataset.upper()} - {model_label}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(CM_DIR, f"{dataset.upper()}_{model_label}.png"), dpi=150)
            plt.close()

        # Class-wise metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
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

total_end_time = time.time()
total_duration = total_end_time - total_start_time
log_print(f"\nTotal Experiment Duration: {total_duration:.2f} seconds")

# -------------------------
# Results Save
# -------------------------
summary_df = pd.DataFrame(summary_results)
classwise_df = pd.DataFrame(classwise_results)
time_df = pd.DataFrame(time_results)  # [추가] 시간 데이터프레임

summary_df.to_csv(os.path.join(BASE_RESULT_DIR, "summary.csv"), index=False)
classwise_df.to_csv(os.path.join(BASE_RESULT_DIR, "classwise.csv"), index=False)
time_df.to_csv(os.path.join(BASE_RESULT_DIR, "execution_times.csv"), index=False)  # [추가] 파일 저장

log_print("\n=== Execution Time Summary ===")
log_print(time_df.to_string(index=False))

log_print("\n=== Accuracy Summary ===")
log_print(summary_df.to_string(index=False))

log_file.close()