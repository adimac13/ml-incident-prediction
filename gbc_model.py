from signal_generator import SignalGenerator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

class IncidentModel:
    def __init__(self):
        self.train_dataset = SignalGenerator(seed = 0)
        self.test_dataset = SignalGenerator(seed=3)
        self.gbc = GradientBoostingClassifier(n_estimators=300,
                                     learning_rate=0.05,
                                     random_state=100,
                                     max_features=5)

    def train(self):
        x, y = self.train_dataset.prepare_dataset()
        x = x.reshape(len(x), -1)
        y = y.reshape(len(y), -1)
        self.gbc.fit(x, y)

    def test(self):
        x, y = self.test_dataset.prepare_dataset()
        x = x.reshape(len(x), -1)
        y = y.reshape(len(y), -1)
        pred_y = self.gbc.predict(x)
        roc = roc_auc_score(y, pred_y)
        recall = recall_score(y, pred_y)
        precision = precision_score(y, pred_y)

        print(f"ROC AUC: {roc:.2f}, RECALL: {recall}, PRECISION: {precision}")

        cm = confusion_matrix(y, pred_y)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=[0,1])
        cm_display.plot()
        plt.show()


if __name__ == "__main__":
    im = IncidentModel()
    im.train()
    im.test()