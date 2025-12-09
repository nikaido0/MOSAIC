import os
import pickle

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score, \
    average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MOSAIC
from train import data_pre, myDataset, Config


def test_model(opt, device, model, test_dataset, model_path):
    print(f"加载模型权重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    test_probs, test_labels = [], []

    with torch.no_grad():
        for pre_feas, seq_feas, global_feas, labels in tqdm(test_loader, desc="测试"):
            pre_feas = pre_feas.to(device) if pre_feas is not None else None
            seq_feas = seq_feas.to(device) if seq_feas is not None else None
            global_feas = global_feas.to(device) if global_feas is not None else None
            labels = labels.to(device)

            outputs = model(pre_feas, seq_feas, global_feas)
            probs = torch.sigmoid(outputs.squeeze(-1)).cpu().numpy()
            test_probs.extend(probs)
            test_labels.extend(labels.cpu().numpy())

    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    test_preds = (test_probs > 0.5).astype(int)

    acc = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, zero_division=0)
    recall = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds)
    mcc = matthews_corrcoef(test_labels, test_preds)
    auc = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.0
    pr_auc = average_precision_score(test_labels, test_probs)

    print(f"测试结果 (阈值=0.5): ACC={acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, "
          f"F1={f1:.3f}, MCC={mcc:.3f}, AUC={auc:.3f}, PR-AUC={pr_auc:.3f}")

    # 保存结果
    save_dir = opt.sublog_path
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "test_results.pkl"), "wb") as f:
        pickle.dump({
            'scores': test_probs,
            'labels': test_labels,
            'preds': test_preds,
            'metrics': {
                'ACC': acc, 'F1': f1, 'MCC': mcc, 'AUC': auc,
                'Precision': precision, 'Recall': recall, 'PR-AUC': pr_auc
            }
        }, f)
    print(f"测试结果已保存到 {save_dir}")


if __name__ == "__main__":
    # 配置
    opt = Config()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 你的数据文件路径（和训练代码保持一致）
    dataset_config = {
        'test_fasta': '../../data/processed/af_filter/test_af.fasta',
        'test_seq_feas_file': '../../data/features/dna2vec/test_reduced.h5',
        'test_pre_feas_file': '../../data/features/gpn_msa/test.h5',
        'test_global_feas_file': '../../data/features/biological/test.h5',
    }

    # 载入测试数据
    test_pre_feas, test_seq_feas, test_global_feas, test_labels = data_pre(
        dataset_config['test_fasta'],
        dataset_config['test_seq_feas_file'],
        dataset_config['test_pre_feas_file'],
        dataset_config['test_global_feas_file']
    )

    # 记录特征维度
    opt.pre_feas_dim = test_pre_feas.shape[-1] if test_pre_feas is not None else 0
    opt.seq_feas_dim = test_seq_feas.shape[-1] if test_seq_feas is not None else 0
    opt.global_feas_dim = test_global_feas.shape[-1] if test_global_feas is not None else 0

    # 创建测试数据集
    test_dataset = myDataset(test_pre_feas, test_seq_feas, test_global_feas, test_labels)

    # 初始化模型（保持和训练一致的参数）
    model = MOSAIC(
        pre_dim=opt.pre_feas_dim,
        seq_dim=opt.seq_feas_dim,
        global_dim=opt.global_feas_dim,
        hidden_dim=opt.hidden,
        num_transformer_layers=opt.n_transformer,
        dropout=opt.drop
    )

    # 固定模型权重路径
    model_path = '../../experiments/exp_train/final/best_model.pth'

    # 日志保存路径（测试结果保存）
    opt.sublog_path = '../../experiments/exp_train/logs'

    test_model(opt, device, model, test_dataset, model_path)
