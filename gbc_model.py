from signal_generator import SignalGenerator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

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
        acc = roc_auc_score(y, pred_y)
        print("Gradient Boosting Classifier ROC AUC score is : {:.2f}".format(acc))

if __name__ == "__main__":
    im = IncidentModel()
    im.train()
    im.test()