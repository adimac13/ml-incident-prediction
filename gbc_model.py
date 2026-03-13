from signal_generator import SignalGenerator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    ds = SignalGenerator(seed = 0)
    ds_t = SignalGenerator(seed=3)

    # Data for training
    x,y = ds.prepare_dataset()
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)

    # Data for testing
    x_t,y_t = ds_t.prepare_dataset()
    x_t = x_t.reshape(len(x_t), -1)
    y_t = y_t.reshape(len(y_t), -1)



    gbc = GradientBoostingClassifier(n_estimators=300,
                                     learning_rate=0.05,
                                     random_state=100,
                                     max_features=5)
    gbc.fit(x, y)

    pred_y = gbc.predict(x_t)

    acc = roc_auc_score(y_t, pred_y)
    print("Gradient Boosting Classifier accuracy is : {:.2f}".format(acc))