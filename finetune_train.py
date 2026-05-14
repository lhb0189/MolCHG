import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from finetune_model import (
    FinetuneModel, FinetuneDataset, finetune_collate_fn,
    load_pretrained_encoder
)
from tools import set_log
from sklearn.metrics import roc_auc_score, mean_squared_error


TASK_NAME = "BBBP"
TASK_TYPE = "classification"
NUM_TASKS = 1
PT_PATH = f"finetune_data/{TASK_NAME}_molgraph.pt"


INPUT_DIM = 15
HIDDEN_DIM = 300
NUM_LAYERS = 5
DROPOUT = 0.5
JK = "sum"
TRAIN_EPS = True
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
PRETRAIN_PATH = "best_pretrain_model.pth"


SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, criterion, device):

    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            node_type = batch["node_type"].to(device)
            mol_scope = batch["mol_scope"].to(device)
            labels = batch["labels"].to(device)

            pred = model(x, edge_index, node_type, mol_scope)

            if TASK_TYPE == "classification":
                valid_mask = (labels != -1)
                if valid_mask.any():
                    loss_matrix = criterion(pred, labels.float())
                    loss = (loss_matrix * valid_mask.float()).sum() / valid_mask.float().sum()
                    total_loss += loss.item()
                    num_batches += 1
            else:
                loss = criterion(pred.squeeze(), labels.squeeze().float())
                total_loss += loss.item()
                num_batches += 1

            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if TASK_TYPE == "classification":
        if NUM_TASKS == 1:
            valid = (all_labels.view(-1) != -1)
            if valid.any():
                score = roc_auc_score(
                    all_labels.view(-1)[valid].numpy(),
                    torch.sigmoid(all_preds.view(-1)[valid]).numpy()
                )
            else:
                score = 0.0
        else:
            auc_list = []
            for t in range(NUM_TASKS):
                valid = (all_labels[:, t] != -1)
                if valid.sum() > 0:
                    try:
                        auc = roc_auc_score(
                            all_labels[:, t][valid].numpy(),
                            torch.sigmoid(all_preds[:, t][valid]).numpy()
                        )
                        auc_list.append(auc)
                    except ValueError:
                        pass
            score = np.mean(auc_list) if auc_list else 0.0
        metric_name = "ROC-AUC"
    else:
        score = mean_squared_error(
            all_labels.squeeze().numpy(),
            all_preds.squeeze().numpy(),
            squared=False
        )
        metric_name = "RMSE"

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, score, metric_name


def run_single_seed(seed, dataset, log):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"{'=' * 60}")
    log.info(f"Seed {seed} Train")
    log.info(f"{'=' * 60}")


    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=finetune_collate_fn, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=finetune_collate_fn, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=finetune_collate_fn, num_workers=0)

    model = FinetuneModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, JK, TRAIN_EPS, NUM_TASKS)
    model = load_pretrained_encoder(model, PRETRAIN_PATH)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if TASK_TYPE == "classification":
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    else:
        criterion = nn.MSELoss()


    best_val_score = -float("inf") if TASK_TYPE == "classification" else float("inf")
    best_test_score = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            node_type = batch["node_type"].to(device)
            mol_scope = batch["mol_scope"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            pred = model(x, edge_index, node_type, mol_scope)

            if TASK_TYPE == "classification":
                valid_mask = (labels != -1)
                if not valid_mask.any():
                    continue
                loss_matrix = criterion(pred, labels.float())
                loss = (loss_matrix * valid_mask.float()).sum() / valid_mask.float().sum()
            else:
                loss = criterion(pred.squeeze(), labels.squeeze().float())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        train_loss /= max(num_batches, 1)

        val_loss, val_score, metric_name = evaluate(model, val_loader, criterion, device)
        _, test_score, _ = evaluate(model, test_loader, criterion, device)


        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(
                f"  [Seed {seed}] Epoch [{epoch + 1}/{EPOCHS}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val {metric_name}: {val_score:.4f} | "
                f"Test {metric_name}: {test_score:.4f}"
            )

        if TASK_TYPE == "classification":
            is_better = val_score > best_val_score
        else:
            is_better = val_score < best_val_score

        if is_better:
            best_val_score = val_score
            best_test_score = test_score
            torch.save(model.state_dict(), f"best_finetune_{TASK_NAME}_seed{seed}.pth")

    log.info(
        f"  [Seed {seed}] 训练完成 | "
        f"Best Val {metric_name}: {best_val_score:.4f} | "
        f"Best Test {metric_name}: {best_test_score:.4f}"
    )
    return best_test_score



if __name__ == "__main__":
    log_save_path = f"finetune_logs/{TASK_NAME}"
    log = set_log(f"finetune_{TASK_NAME}", log_save_path)

    dataset = FinetuneDataset(PT_PATH)
    metric_name = "ROC-AUC" if TASK_TYPE == "classification" else "RMSE"

    log.info(f"Task: {TASK_NAME} | Type: {TASK_TYPE} | Metric: {metric_name}")
    log.info(f"Dataset size: {len(dataset)} | Seeds: {SEEDS}")
    log.info(f"Model: GIN {NUM_LAYERS}L, hidden={HIDDEN_DIM}, JK={JK}, dropout={DROPOUT}")
    log.info(f"Training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}")

    all_test_scores = []

    for seed in SEEDS:
        score = run_single_seed(seed, dataset, log)
        all_test_scores.append(score)

    mean_score = np.mean(all_test_scores)
    std_score = np.std(all_test_scores)

    log.info(f"{'=' * 60}")
    log.info(f"{'=' * 60}")
    log.info(f"  {TASK_NAME} 最终结果 ({len(SEEDS)} seeds)")
    log.info(f"  每个 seed 的 Test {metric_name}:")
    for i, (seed, score) in enumerate(zip(SEEDS, all_test_scores)):
        log.info(f"    Seed {seed}: {score:.4f}")
    log.info(f"  -----------------------------------------")
    log.info(f"  Mean ± Std: {mean_score:.4f} ± {std_score:.4f}")
    log.info(f"{'=' * 60}")