import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/wangzhou/leaves/logs/resnet18/train-2025-07-19/01-51-20/logs/version_0/metrics.csv')
df_loss = df.dropna(subset=['train_loss_epoch', 'val_loss'])

# 画图
plt.figure(figsize=(10, 6))
plt.plot(df_loss['epoch'], df_loss['train_loss_epoch'], label='Train Loss', marker='o')
plt.plot(df_loss['epoch'], df_loss['val_loss'], label='Validation Loss', marker='x')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()
