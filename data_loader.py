import numpy as np
import pandas as pd

class DataLoader(object):

    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size

        self.indices = np.arange(self.X.shape[0])
        np.random.seed(1)

    def shuffle(self):
        np.random.shuffle(self.indices)

    def get_batch(self, mode='train'):
        X_batch = []
        y_batch = []

        if mode == 'train':
            self.shuffle()

        for i in range(0, len(self.indices), self.batch_size):
            indices = self.indices[i:i + self.batch_size]

            # Handle both Pandas and Numpy data types
            if isinstance(self.X, pd.DataFrame) or isinstance(self.X, pd.Series):
                X_batch.append(self.X.iloc[indices])
                y_batch.append(self.y.iloc[indices])
            else:  # Assuming NumPy array
                X_batch.append(self.X[indices])
                y_batch.append(self.y[indices])

        return X_batch, y_batch
