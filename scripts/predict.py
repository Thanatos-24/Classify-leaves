from dataloader import preprocess_image_label_for_test, preprocess_label
from dataprocess import train_valid_dataset, test_dataset
import torch
from torch.utils.data import DataLoader, Dataset
import sys
import os
import csv
import hydra
import json
import pytorch_lightning as pl
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from net.Resnet18 import ResNet

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def test(config):
    config = config["config"]
    ckpt_path = config.ckpt 
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResNet.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to(device)
    
    with open(config.Datasetconfig.mapping_path, 'r') as f:
        mapping = json.load(f)

    idx_to_name = {v: k for k, v in mapping.items()}
    test_image = preprocess_image_label_for_test(config.Datasetconfig.test_image_root, sigmaX=config.Datasetconfig.sigmaX)
    test_ds = test_dataset(test_image)
    print("loaded data,start test")

    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    test_preds = []

    with torch.no_grad():
        for batch in tqdm(test_dl):
            if isinstance(batch, (list, tuple)):
                imgs, paths = batch
            else:
                raise RuntimeError("DataLoader 返回内容需要是(img_tensor, img_path)格式")
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            for path, pred in zip(paths, preds.cpu().numpy()):
                filename = os.path.basename(path)  # 提取文件名 18353.jpg
                label_name = idx_to_name[pred]  # json中的key是字符串
                test_preds.append((f"images/{filename}", label_name))

    # === 4. 保存为 CSV ===
    os.makedirs(config.output_dir, exist_ok=True)
    save_path = os.path.join(config.output_dir, "test_predictions.csv")
    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for row in test_preds:
            writer.writerow(row)

    print(f"测试完成，预测结果保存在: {save_path}")

if __name__ == "__main__":
    test()
