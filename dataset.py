import numpy as np
from matplotlib import pyplot as plt

class Dataset:
    """
    This class prepares a dataset for training. It generates a linearly increasing sine function with noise and incident
    applied.
    """
    def __init__(self, freq = 2, time_range = 20, num_of_probes = 10000):
        self.freq = freq
        self.time_range = time_range
        self.num_of_probes = num_of_probes
        self.where_incident = np.zeros(self.num_of_probes)
        self.signal = None

    def setup(self):
        """
        Setup function. It returns generated signal and labels, where: 0 -> normal, 1 -> incident.
        """
        time = np.linspace(0, self.time_range, self.num_of_probes)
        noise = np.random.normal(0, 0.1, self.num_of_probes) / 10
        sine = np.sin(time * self.freq) / 10
        linear = time / 10

        self.signal = noise + sine + linear
        i = 0
        while i < self.num_of_probes - 1:
            gap = int(np.random.exponential()*700)
            i += gap
            len_of_incident = np.random.randint(50, 100)
            self.signal[i : (i+len_of_incident)] += 0.2
            self.where_incident[i : (i+len_of_incident)] = 1

            i += len_of_incident # To avoid incident overlapping

        if __name__ == "__main__":
            plt.plot(time, self.signal)
            plt.show()
        return self.signal, self.where_incident

if __name__ == "__main__":
    ds = Dataset()
    ds.setup()
    # pass