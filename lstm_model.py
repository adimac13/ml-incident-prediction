from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from signal_generator import SignalGenerator
from torch import nn, optim, tensor
import torch
import torchmetrics

class IncidentDataset(Dataset):
    def __init__(self, current_window_size = 200, future_window_size = 100, freq = 2, time_range = 20, num_of_probes = 10000):
        sg = SignalGenerator(future_window_size, current_window_size, freq, time_range, num_of_probes)
        self.input, self.labels = sg.prepare_dataset()

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return tensor(self.input[idx], dtype = torch.float32), tensor(self.labels[idx], dtype = torch.float32)

class IncidentDataModule(pl.LightningDataModule):
    def __init__(self, batch_size = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage = None):
        ds = IncidentDataset()
        self.ds = ds
        train_ds_len = len(ds) - len(ds) // 10
        self.train_dataset = Subset(ds, range(0,train_ds_len))
        self.val_dataset = Subset(ds, range(train_ds_len,len(ds)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)

class IncidentModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size=128, num_layers=2, batch_first = True, dropout = 0.2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu  = nn.ReLU()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))

        self.acc_train = torchmetrics.Accuracy(task='binary')
        self.precision_train  = torchmetrics.Precision(task='binary')
        self.recall_train  = torchmetrics.Recall(task='binary')
        self.auroc_train  = torchmetrics.AUROC(task="binary")

        self.acc_val = torchmetrics.Accuracy(task='binary')
        self.precision_val = torchmetrics.Precision(task='binary')
        self.recall_val = torchmetrics.Recall(task='binary')
        self.auroc_val = torchmetrics.AUROC(task="binary")


        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_final = lstm_out[:,-1,:]
        out = self.fc1(lstm_final)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = 1e-4)
        return optimizer

    def training_step(self, batch):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)

        outputs = self.sigmoid(outputs)
        acc = self.acc_train(outputs, labels)
        precision = self.precision_train(outputs, labels)
        recall = self.recall_train(outputs, labels)
        auroc = self.auroc_train(outputs, labels)
        self.log('train_loss', loss, on_step = True, on_epoch = True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_precision', precision, on_step=False, on_epoch=True)
        self.log('train_recall', recall, on_step=False, on_epoch=True)
        self.log('train_auroc', auroc, on_step = False, on_epoch = True)

        return loss

    def validation_step(self, batch):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)

        outputs = self.sigmoid(outputs)
        acc = self.acc_val(outputs, labels)
        precision = self.precision_val(outputs, labels)
        recall = self.recall_val(outputs, labels)
        auroc = self.auroc_val(outputs, labels)
        self.log('val_loss', loss, on_step = True, on_epoch = True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True)
        self.log('val_auroc', auroc, on_step=False, on_epoch=True)

        return loss

if __name__ == "__main__":

    # ds = IncidentDataset()
    # a = sum(ds[:][1])
    # b = len(ds[:][1])
    # print(b/a)
    idm = IncidentDataModule()
    idm.setup()
    tdt = idm.train_dataloader()
    features, labels = next(iter(tdt))
    print(features.shape)

    incident_model = IncidentModel()
    logger = TensorBoardLogger('./logs', name = 'incident-model')
    trainer = pl.Trainer(logger = logger, max_epochs = 20, accelerator='gpu')
    trainer.fit(incident_model, idm)