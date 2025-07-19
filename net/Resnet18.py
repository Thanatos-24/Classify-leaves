import torch
import torch.nn as nn
import pytorch_lightning as pl

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride) if use_1conv else None

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        identity = self.conv3(x) if self.conv3 else x
        y += identity
        return self.relu(y)

class ResNet(pl.LightningModule):
    def __init__(self, input_channels, num_channels, num_residuals, num_classes, learning_rate):
        super(ResNet, self).__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(num_channels, num_channels, num_residuals[0], first_block=True)
        self.layer2 = self._make_layer(num_channels, num_channels * 2, num_residuals[1])
        self.layer3 = self._make_layer(num_channels * 2, num_channels * 4, num_residuals[2])
        self.layer4 = self._make_layer(num_channels * 4, num_channels * 8, num_residuals[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_channels * 8, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def _make_layer(self, input_channels, output_channels, num_blocks, first_block=False):
        layers = []
        for i in range(num_blocks):
            if i == 0 and not first_block:
                layers.append(Residual(input_channels, output_channels, use_1conv=True, stride=2))
            else:
                layers.append(Residual(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

# 测试 forward 是否正常运行
if __name__ == "__main__":
    model = ResNet18(input_channels=3, num_channels=64, num_classes=176)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)  # Expected: torch.Size([1, 10])
