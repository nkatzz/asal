import numpy as np


class SAXTransformer:

    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size

    @staticmethod
    def normalize_series(series):
        """Normalize the series to zero mean and unit variance."""
        return (series - series.mean()) / series.std()

    @staticmethod
    def sax_transform_value(value, breakpoints):
        """Transform a single value using Symbolic Aggregate Approximation."""
        for i in range(len(breakpoints)):
            if value < breakpoints[i]:
                return i
        return len(breakpoints)

    def column_sax_transform(self, column):
        """Transform an entire column (time series) using Symbolic Aggregate Approximation with handling for constant
        sequences."""
        # If the column is constant, map to the median symbol
        if np.std(column) == 0:
            return [self.alphabet_size // 2] * len(column)

        # Otherwise, proceed with the usual SAX transformation
        breakpoints = np.percentile(column, np.linspace(0, 100, self.alphabet_size + 1)[1:-1])
        normalized_column = self.normalize_series(column)
        return [self.sax_transform_value(val, breakpoints) for val in normalized_column]
