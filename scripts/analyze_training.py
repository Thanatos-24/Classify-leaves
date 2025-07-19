import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def analyze_training_logs(log_path):
    """分析训练日志"""
    try:
        df = pd.read_csv(log_path)
        print("=== 训练分析报告 ===")
        print(f"总训练步数: {len(df)}")
        print(f"总训练轮数: {df['epoch'].max() + 1}")
        
        # 分析损失变化
        if 'train_loss_epoch' in df.columns:
            train_loss = df['train_loss_epoch'].dropna()
            print(f"训练损失范围: {train_loss.min():.4f} - {train_loss.max():.4f}")
            print(f"最终训练损失: {train_loss.iloc[-1]:.4f}")
        
        if 'val_loss' in df.columns:
            val_loss = df['val_loss'].dropna()
            print(f"验证损失范围: {val_loss.min():.4f} - {val_loss.max():.4f}")
            print(f"最终验证损失: {val_loss.iloc[-1]:.4f}")
        
        # 分析准确率
        if 'train_acc_epoch' in df.columns:
            train_acc = df['train_acc_epoch'].dropna()
            print(f"训练准确率范围: {train_acc.min():.4f} - {train_acc.max():.4f}")
            print(f"最终训练准确率: {train_acc.iloc[-1]:.4f}")
        
        if 'val_acc' in df.columns:
            val_acc = df['val_acc'].dropna()
            print(f"验证准确率范围: {val_acc.min():.4f} - {val_acc.max():.4f}")
            print(f"最终验证准确率: {val_acc.iloc[-1]:.4f}")
        
        # 绘制训练曲线
        plt.figure(figsize=(15, 10))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        if 'train_loss_epoch' in df.columns:
            train_loss_data = df[['epoch', 'train_loss_epoch']].dropna()
            if len(train_loss_data) > 0:
                plt.plot(train_loss_data['epoch'], train_loss_data['train_loss_epoch'], label='Train Loss')
        if 'val_loss' in df.columns:
            val_loss_data = df[['epoch', 'val_loss']].dropna()
            if len(val_loss_data) > 0:
                plt.plot(val_loss_data['epoch'], val_loss_data['val_loss'], label='Val Loss')
        plt.title('Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 准确率曲线
        plt.subplot(2, 2, 2)
        if 'train_acc_epoch' in df.columns:
            train_acc_data = df[['epoch', 'train_acc_epoch']].dropna()
            if len(train_acc_data) > 0:
                plt.plot(train_acc_data['epoch'], train_acc_data['train_acc_epoch'], label='Train Acc')
        if 'val_acc' in df.columns:
            val_acc_data = df[['epoch', 'val_acc']].dropna()
            if len(val_acc_data) > 0:
                plt.plot(val_acc_data['epoch'], val_acc_data['val_acc'], label='Val Acc')
        plt.title('Accuracy vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # 学习率分析（如果有的话）
        plt.subplot(2, 2, 3)
        if 'lr-AdamW' in df.columns:
            lr_data = df[['epoch', 'lr-AdamW']].dropna()
            if len(lr_data) > 0:
                plt.plot(lr_data['epoch'], lr_data['lr-AdamW'])
                plt.title('Learning Rate vs Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.grid(True)
        
        # 梯度分析
        plt.subplot(2, 2, 4)
        if 'train_loss_step' in df.columns:
            step_loss_data = df['train_loss_step'].dropna()
            if len(step_loss_data) > 0:
                plt.plot(step_loss_data.index, step_loss_data.values)
                plt.title('Training Loss vs Step')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 诊断建议
        print("\n=== 诊断建议 ===")
        
        if 'val_loss' in df.columns and 'train_loss_epoch' in df.columns:
            val_loss = df['val_loss'].dropna()
            train_loss = df['train_loss_epoch'].dropna()
            
            if len(val_loss) > 0 and len(train_loss) > 0:
                # 检查过拟合
                if train_loss.iloc[-1] < val_loss.iloc[-1] * 0.8:
                    print("⚠️  检测到过拟合：训练损失远低于验证损失")
                    print("   建议：增加正则化、减少模型复杂度、增加数据增强")
                
                # 检查欠拟合
                if train_loss.iloc[-1] > 2.0:
                    print("⚠️  检测到欠拟合：训练损失过高")
                    print("   建议：增加模型复杂度、降低学习率、增加训练轮数")
                
                # 检查学习率
                if train_loss.iloc[-1] > 4.0:
                    print("⚠️  学习率可能过高：训练损失过高")
                    print("   建议：降低学习率到0.0001或更低")
        
        if 'val_acc' in df.columns:
            val_acc = df['val_acc'].dropna()
            if len(val_acc) > 0 and val_acc.iloc[-1] < 0.1:
                print("⚠️  验证准确率过低（<10%）")
                print("   建议：检查数据预处理、标签质量、模型架构")
        
        print("\n=== 改进建议 ===")
        print("1. 数据预处理：移除过度的图像锐化处理")
        print("2. 数据标准化：添加ImageNet标准化")
        print("3. 数据增强：添加随机翻转、旋转、颜色抖动")
        print("4. 学习率：降低到0.0001")
        print("5. 优化器：使用AdamW + 权重衰减")
        print("6. 学习率调度：添加余弦退火调度器")
        print("7. 早停：添加早停机制防止过拟合")
        print("8. 梯度裁剪：添加梯度裁剪防止梯度爆炸")
        
    except Exception as e:
        print(f"分析日志时出错: {e}")

if __name__ == "__main__":
    # 查找最新的训练日志
    log_dirs = [
        "logs/resnet18/train-2025-07-19/02-05-36/logs/version_0/metrics.csv",
        "logs/alexnet/train-2025-07-19/02-05-36/logs/version_0/metrics.csv"
    ]
    
    for log_path in log_dirs:
        if os.path.exists(log_path):
            print(f"分析日志: {log_path}")
            analyze_training_logs(log_path)
            break
    else:
        print("未找到训练日志文件") 