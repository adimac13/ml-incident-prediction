import numpy as np
from matplotlib import pyplot as plt

class SignalGenerator:
    """
    This class prepares a dataset for training. It generates a linearly increasing sine function with noise and incident
    applied.
    """
    def __init__(self, freq = 2, time_range = 20, num_of_probes = 10000, next_H_steps = 20, prev_W_steps = 20):
        self.freq = freq
        self.time_range = time_range
        self.num_of_probes = num_of_probes
        self.where_incident = np.zeros(self.num_of_probes)
        self.signal = None
        self.H_steps = next_H_steps
        self.W_steps = prev_W_steps

    def setup(self):
        """
        Setup function. Function creates signal and labels, where: 0 -> normal, 1 -> incident.
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

    def prepare_dataset(self):
        """
        Function returns windows (consisting of W steps) and label whether in next H steps incident will occur.
        """
        self.setup()
        xs = []
        ys = []
        for i in range(len(self.signal) - self.W_steps - 1):
            # W step window
            x = self.signal[i:i + self.W_steps]

            # H step window, after W step window
            max_y_idx = min(len(self.signal) - 1, i + self.W_steps + self.H_steps)
            y = (1 if 1 in self.where_incident[i + self.W_steps : max_y_idx + 1] else 0)

            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

if __name__ == "__main__":
    ds = SignalGenerator()
    x,y = ds.prepare_dataset()
    print(x.shape, y.shape)
    print(np.sum(y))
    # pass