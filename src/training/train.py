import math
import datetime
import glob
import os
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score, \
    average_precision_score

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from torchnet import meter
from src.utils.preprocess import parse_fasta


# 配置类
class Config:
    def __init__(self):
        self.batch_size = 256
        self.hidden = 256
        self.n_transformer = 3
        self.drop = 0.2
        self.epoch = 100
        self.lr = 1e-4
        self.seed = 42
        self.max_metric = 'AUC'
        self.early_stop_epochs = 5
        self.pre_feas_dim = 0
        self.seq_feas_dim = 0
        self.global_feas_dim = 0  # 保留字段但不再使用
        self.submodel_path = ''
        self.sublog_path = ''
        self.valid_split = 0.2

    def print_config(self):
        print("模型配置:")
        for name, value in vars(self).items():
            print(f' {name} = {value}')


# 日志类
class Logger:
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if not self.log.closed:
            self.log.close()


# 特征提取函数
def get_features(names: list, sequences: list, feature_file: str) -> np.ndarray:
    if names is None or len(names) == 0:
        raise ValueError("names 列表不能为空")

    print(f"开始从 {os.path.basename(feature_file)} 提取特征...")
    pure_ids = [name.split("|")[0] for name in names]
    features_list = []
    expected_shape = None

    try:
        with h5py.File(feature_file, "r") as h5fi:
            for pid in tqdm(pure_ids, desc=f"加载特征 from {os.path.basename(feature_file)}"):
                if pid not in h5fi:
                    raise KeyError(f"HDF5 文件中未找到 ID: {pid}")

                feature_data = h5fi[pid][:]
                if expected_shape is None:
                    expected_shape = feature_data.shape
                elif feature_data.shape != expected_shape:
                    raise ValueError(
                        f"ID {pid} 的特征形状 {feature_data.shape} 与预期形状 {expected_shape} 不一致")

                features_list.append(feature_data)

        features_array = np.stack(features_list, axis=0)
        print(f"特征提取完成, 形状: {features_array.shape}")
        return features_array

    except Exception as e:
        raise RuntimeError(f"处理特征文件 {feature_file} 时出错: {str(e)}")


# 数据预处理 - 只处理 pre_feas 和 seq_feas，global_feas 永远返回 None
def data_pre(fasta_file, seq_feas_file=None, pre_feas_file=None, global_feas_file=None):
    names, sequences, labels = parse_fasta(fasta_file)
    labels = np.array(labels)
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("标签必须为 0 或 1")

    seq_feas = None
    if seq_feas_file:
        try:
            seq_feas = get_features(names, sequences, seq_feas_file)
            if np.any(np.isnan(seq_feas)) or np.any(np.isinf(seq_feas)):
                print("警告: 序列特征包含 NaN 或 Inf 值，进行清理")
                seq_feas = np.nan_to_num(seq_feas, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"加载序列特征文件 {seq_feas_file} 失败: {str(e)}")
            raise

    pre_feas = None
    if pre_feas_file:
        try:
            pre_feas = get_features(names, sequences, pre_feas_file)
            if np.any(np.isnan(pre_feas)) or np.any(np.isinf(pre_feas)):
                print("警告: 预训练特征包含 NaN 或 Inf 值，进行清理")
                pre_feas = np.nan_to_num(pre_feas, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"加载预训练特征文件 {pre_feas_file} 失败: {str(e)}")
            raise

    # global_feas 永远返回 None
    global_feas = None

    return pre_feas, seq_feas, global_feas, labels


# 数据分割辅助函数
def split_data(data, indices):
    return data[indices] if data is not None else None


# 数据集类 - 已移除 global_feas
class myDataset(Dataset):
    def __init__(self, pre_feas, seq_feas, labels):
        self.pre_feas = torch.tensor(pre_feas, dtype=torch.float32) if pre_feas is not None else None
        self.seq_feas = torch.tensor(seq_feas, dtype=torch.float32) if seq_feas is not None else None
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, index):
        pre = self.pre_feas[index] if self.pre_feas is not None else None
        seq = self.seq_feas[index] if self.seq_feas is not None else None
        return pre, seq, self.labels[index]  # 干净利落！

    def __len__(self):
        return len(self.labels)


# 训练函数 - 传入的 global_feas 永远是 None
def train(opt, device, model, train_data, valid_data, fold=None):
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=False, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.to(device)

    loss_meter = meter.AverageValueMeter()
    best_auc = 0.0
    early_stop_iter = 0
    best_valid_preds = None
    best_valid_targets = None

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    fold_str = "Single Split" if fold is None else f"Fold {fold + 1}"
    print(f"{fold_str} 开始训练...")
    for epoch in range(opt.epoch):
        model.train()
        loss_meter.reset()
        train_preds, train_targets = [], []

        for pre_feas, seq_feas, target in tqdm(train_dataloader,
                                               desc=f"{fold_str} Epoch {epoch + 1}/{opt.epoch} [训练]"):
            pre_feas = pre_feas.to(device) if pre_feas is not None else None
            seq_feas = seq_feas.to(device) if seq_feas is not None else None
            target = target.to(device)

            optimizer.zero_grad()
            output = model(pre_feas, seq_feas)  # global_feas 传 None
            score = output.squeeze(-1)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            probs = torch.sigmoid(score).cpu().detach().numpy()
            train_preds.extend(probs)
            train_targets.extend(target.cpu().detach().numpy())

        train_preds = np.array(train_preds)
        train_preds_binary = (train_preds > 0.5).astype(int)
        train_acc = accuracy_score(train_targets, train_preds_binary)
        train_losses.append(loss_meter.mean)
        train_accs.append(train_acc)

        model.eval()
        val_loss_meter = meter.AverageValueMeter()
        valid_preds, valid_targets = [], []

        with torch.no_grad():
            for pre_feas, seq_feas, target in tqdm(valid_dataloader,
                                                   desc=f"{fold_str} Epoch {epoch + 1}/{opt.epoch} [验证]"):
                pre_feas = pre_feas.to(device) if pre_feas is not None else None
                seq_feas = seq_feas.to(device) if seq_feas is not None else None
                target = target.to(device)

                output = model(pre_feas, seq_feas)
                score = output.squeeze(-1)
                val_loss = criterion(score, target)
                val_loss_meter.add(val_loss.item())
                probs = torch.sigmoid(score).cpu().numpy()
                valid_preds.extend(probs)
                valid_targets.extend(target.cpu().numpy())

        valid_preds = np.array(valid_preds)
        valid_targets = np.array(valid_targets)

        valid_preds_binary = (valid_preds > 0.5).astype(int)
        acc = accuracy_score(valid_targets, valid_preds_binary)
        precision = precision_score(valid_targets, valid_preds_binary, zero_division=0)
        recall = recall_score(valid_targets, valid_preds_binary, zero_division=0)
        f1 = f1_score(valid_targets, valid_preds_binary)
        mcc = matthews_corrcoef(valid_targets, valid_preds_binary)
        auc = roc_auc_score(valid_targets, valid_preds) if len(np.unique(valid_targets)) > 1 else 0.0
        pr_auc = average_precision_score(valid_targets, valid_preds)

        val_losses.append(val_loss_meter.mean)
        val_accs.append(acc)

        print(
            f"{fold_str} Epoch {epoch + 1}: 训练损失 = {loss_meter.mean:.5f}, 验证损失 = {val_loss_meter.mean:.5f}")
        print(f"验证指标 (阈值=0.5): ACC={acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, "
              f"F1={f1:.3f}, MCC={mcc:.3f}, AUC={auc:.3f}, PR-AUC={pr_auc:.3f}")

        if auc > best_auc:
            best_auc = auc
            best_valid_preds = valid_preds
            best_valid_targets = valid_targets
            model_path = os.path.join(opt.submodel_path, 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
            }, model_path)
            print(f"保存当前最佳模型: {model_path} (AUC={auc:.3f})")
            early_stop_iter = 0
        else:
            early_stop_iter += 1
            if early_stop_iter >= opt.early_stop_epochs:
                print(f"{fold_str} 早停触发: {early_stop_iter} 轮未提升")
                break

        scheduler.step(auc)
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title(f'{fold_str} Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.title(f'{fold_str} Training and Validation Accuracy')

    plt.tight_layout()
    plot_path = os.path.join(opt.sublog_path, f'training_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"训练指标图表已保存到: {plot_path}")

    return model, best_auc, best_valid_preds, best_valid_targets


# 测试函数
def test(opt, device, model, test_data):
    model_path = os.path.join(opt.submodel_path, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"在 {opt.submodel_path} 中未找到模型文件: best_model.pth")
    print(f"加载最佳模型: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
    test_probs, test_targets = [], []

    print("测试集评估...")
    with torch.no_grad():
        for pre_feas, seq_feas, target in tqdm(test_dataloader, desc="测试"):
            pre_feas = pre_feas.to(device) if pre_feas is not None else None
            seq_feas = seq_feas.to(device) if seq_feas is not None else None
            target = target.to(device)

            output = model(pre_feas, seq_feas)
            score = output.squeeze(-1)
            probs = torch.sigmoid(score).cpu().numpy()
            test_probs.extend(probs)
            test_targets.extend(target.cpu().numpy())

    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    test_preds = (test_probs > 0.5).astype(int)

    acc = accuracy_score(test_targets, test_preds)
    precision = precision_score(test_targets, test_preds, zero_division=0)
    recall = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_preds, test_targets)
    mcc = matthews_corrcoef(test_targets, test_preds)
    auc = roc_auc_score(test_targets, test_probs) if len(np.unique(test_targets)) > 1 else 0.0
    pr_auc = average_precision_score(test_targets, test_probs)

    print(f"测试结果 (阈值=0.5): ACC={acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, "
          f"F1={f1:.3f}, MCC={mcc:.3f}, AUC={auc:.3f}, PR-AUC={pr_auc:.3f}")

    results = {
        'probs': test_probs,
        'targets': test_targets,
        'preds': test_preds,
        'metrics': {'ACC': acc, 'F1': f1, 'MCC': mcc, 'AUC': auc},
        'threshold': 0.5
    }
    results_path = os.path.join(opt.sublog_path, 'test_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    metrics_path = os.path.join(opt.sublog_path, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Threshold: 0.5\n")
        f.write(f"ACC: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")

    print(f"测试结果已保存到: {results_path}")
    print(f"测试指标已保存到: {metrics_path}")

    return results


# 主函数
def main():
    opt = Config()
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    dataset_config = {
        'train_fasta': '../../data/raw/fasta/train_sequences_alt_labeled.fasta',
        'test_fasta': '../../data/raw/fasta/test_sequences_alt_labeled.fasta',
        'train_seq_feas_file': '../../data/features/dna2vec/PCA/train_alt.h5',
        'test_seq_feas_file': '../../data/features/dna2vec/PCA/test_alt.h5',
        'train_pre_feas_file': '../../data/features/gpn_msa/PCA/train_alt.h5',
        'test_pre_feas_file': '../../data/features/gpn_msa/PCA/test_alt.h5',
        # global_feas 文件即使存在也完全忽略
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_{timestamp}"
    base_exp_dir = os.path.join('../../experiments', exp_name)

    opt.submodel_path = os.path.join(base_exp_dir, 'checkpoints')
    opt.sublog_path = os.path.join(base_exp_dir, 'logs')

    os.makedirs(opt.submodel_path, exist_ok=True)
    os.makedirs(opt.sublog_path, exist_ok=True)

    sys.stdout = Logger(os.path.join(opt.sublog_path, 'training.log'))
    opt.print_config()

    try:
        print("加载训练数据...")
        train_pre_feas, train_seq_feas, _, train_labels = data_pre(
            dataset_config['train_fasta'],
            dataset_config['train_seq_feas_file'],
            dataset_config['train_pre_feas_file']
        )

        if train_pre_feas is not None:
            opt.pre_feas_dim = train_pre_feas.shape[-1]
            print(f"预训练特征维度: {opt.pre_feas_dim}")
        if train_seq_feas is not None:
            opt.seq_feas_dim = train_seq_feas.shape[-1]
            print(f"序列特征维度: {opt.seq_feas_dim}")

        print("加载测试数据...")
        test_pre_feas, test_seq_feas, _, test_labels = data_pre(
            dataset_config['test_fasta'],
            dataset_config['test_seq_feas_file'],
            dataset_config['test_pre_feas_file']
        )

        train_names, _, _ = parse_fasta(dataset_config['train_fasta'])
        test_names, _, _ = parse_fasta(dataset_config['test_fasta'])
        common_ids = set(train_names).intersection(set(test_names))
        if common_ids:
            print(f"警告: 训练集和测试集存在 {len(common_ids)} 个重复 ID")

        from model import MOSAIC_wo_global
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)

        train_idx, val_idx = train_test_split(
            np.arange(len(train_labels)),
            test_size=opt.valid_split,
            random_state=opt.seed,
            stratify=train_labels
        )

        print(f"训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}")
        print(f"训练集正样本比例: {np.mean(train_labels[train_idx]):.3f}")
        print(f"验证集正样本比例: {np.mean(train_labels[val_idx]):.3f}")

        # 分割特征
        train_pre_feas_split = split_data(train_pre_feas, train_idx)
        val_pre_feas_split = split_data(train_pre_feas, val_idx)
        train_seq_feas_split = split_data(train_seq_feas, train_idx)
        val_seq_feas_split = split_data(train_seq_feas, val_idx)

        # 创建数据集（不再传 global_feas）
        train_dataset = myDataset(train_pre_feas_split, train_seq_feas_split, train_labels[train_idx])
        val_dataset = myDataset(val_pre_feas_split, val_seq_feas_split, train_labels[val_idx])
        test_dataset = myDataset(test_pre_feas, test_seq_feas, test_labels)

        # 初始化模型（MOSAIC_wo_global 本来就不需要 global_feas）
        model = MOSAIC_wo_global(
            pre_dim=opt.pre_feas_dim,
            seq_dim=opt.seq_feas_dim,
            hidden_dim=opt.hidden,
            num_transformer_layers=opt.n_transformer,
            dropout=opt.drop
        )

        # 训练 + 测试
        model, best_auc, _, _ = train(opt, device, model, train_dataset, val_dataset)
        test_results = test(opt, device, model, test_dataset)

    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
