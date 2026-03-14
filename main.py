from lstm_model import IncidentModel_train
from gbc_model import IncidentModel as IncidentModel_gbc

if __name__ == "__main__":
    # To train lstm model
    # IncidentModel_train()

    # To train and evaluate gbc model
    im = IncidentModel_gbc()
    im.train()
    im.test()