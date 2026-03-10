from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from signal_generator import SignalGenerator
from torch import nn, optim, tensor
import torch
import torchmetrics

class IncidentDataset(Dataset):
    def __init__(self, current_window_size = 20, future_window_size = 20, freq = 2, time_range = 20, num_of_probes = 10000):
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
        self.val_dataset, self.train_dataset = random_split(ds, [len(ds) // 10, len(ds) - len(ds) // 10])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)

class IncidentModel(pl.LightningModule):
    def __init__(self, hidden_dim = 30):
        super().__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size=hidden_dim, num_layers=1, batch_first = True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, 1)
        self.loss = nn.BCEWithLogitsLoss()
        self.acc = torchmetrics.Accuracy(task='binary')
        self.precision = torchmetrics.Precision(task='binary')
        self.recall = torchmetrics.Recall(task='binary')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_final = lstm_out[:,-1,:]
        out = self.fc(lstm_final)
        return out

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = 1e-4)
        return optimizer

    def training_step(self, batch):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)

        outputs = self.sigmoid(outputs)
        acc = self.acc(outputs, labels)
        precision = self.precision(outputs, labels)
        recall = self.recall(outputs, labels)
        self.log('train_loss', loss, on_step = True, on_epoch = True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_precision', precision, on_step=False, on_epoch=True)
        self.log('train_recall', recall, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)

        outputs = self.sigmoid(outputs)
        acc = self.acc(outputs, labels)
        precision = self.precision(outputs, labels)
        recall = self.recall(outputs, labels)
        self.log('val_loss', loss, on_step = True, on_epoch = True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True)

        return loss

if __name__ == "__main__":
    idm = IncidentDataModule()
    idm.setup()

    incident_model = IncidentModel()
    logger = TensorBoardLogger('./logs', name = 'incident-model')
    trainer = pl.Trainer(logger = logger, max_epochs = 10)
    trainer.fit(incident_model, idm)