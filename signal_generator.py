import numpy as np
from matplotlib import pyplot as plt

class SignalGenerator:
    """
    This class prepares a dataset for training. It generates a linearly increasing sine function with noise and incident
    applied.
    """
    def __init__(self, next_H_steps = 100, prev_W_steps = 200, freq = 2, time_range = 20, num_of_probes = 10000, seed = 0):
        self.freq = freq
        self.time_range = time_range
        self.num_of_probes = num_of_probes
        self.where_incident = np.zeros(self.num_of_probes)
        self.signal = None
        self.H_steps = next_H_steps
        self.W_steps = prev_W_steps
        self.seed = seed

    def _setup(self):
        """
        Setup function. Function creates signal and labels, where: 0 -> normal, 1 -> incident.
        """
        np.random.seed(self.seed)
        time = np.linspace(0, self.time_range, self.num_of_probes)
        noise = np.random.normal(0, 0.1, self.num_of_probes) / 20
        sine = np.sin(time * self.freq) / 20
        linear = 0

        self.signal = noise + sine + linear

        pre_incident_len = self.W_steps
        linspace = np.linspace(-2,0,pre_incident_len)
        pre_incident = np.exp(linspace)


        i = pre_incident_len
        while True:
            gap = int(np.random.exponential()*300)
            i += gap
            if i > len(self.signal) - 1:
                break

            len_of_incident = np.random.randint(100, 200)

            self.signal[i : (i+len_of_incident)] += 0.2
            self.signal[i - pre_incident_len : i] += pre_incident * 0.2
            self.where_incident[i: (i+len_of_incident)] = 1

            i += len_of_incident + 800 # To avoid incident overlapping

        # self.signal = (self.signal - min(self.signal)) / (max(self.signal) - min(self.signal))


        # plt.plot(time, self.signal)
        # plt.fill_between(time, plt.ylim()[0], plt.ylim()[1], where=self.where_incident == 1, color='red', alpha=0.3)
        # plt.show()

    def prepare_dataset(self):
        """
        Function returns windows (consisting of W steps) and label whether in next H steps incident will occur.
        """
        self._setup()
        xs = []
        ys = []
        for i in range(0, len(self.signal) - self.W_steps - 1, 5):
            # Getting rid of samples containing incident
            if 1 in self.where_incident[i:i + self.W_steps]:
                continue

            # W step window
            x = self.signal[i:i + self.W_steps]

            # H steps window, after W steps window
            max_y_idx = min(len(self.signal) - 1, i + self.W_steps + self.H_steps)
            y = (1 if 1 in self.where_incident[i + self.W_steps : max_y_idx ] else 0)

            xs.append(x)
            ys.append(y)

        xs = np.array(xs)
        ys = np.array(ys)

        trainX = xs[:,:,np.newaxis]

        trainY = ys[:, np.newaxis]

        return trainX, trainY

if __name__ == "__main__":
    ds = SignalGenerator()
    x,y = ds.prepare_dataset()
    print(y.mean())
    # ds._setup()
    # pass