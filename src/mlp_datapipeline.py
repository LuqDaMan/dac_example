import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataPipeline:
    """Data pipeline for MLP: OneHotEncode categoricals, standardize integers."""

    def __init__(self):
        self.preprocessor = None
        self.label_encoder = None
        self.fitted = False

        # Qualitative (categorical) features - all have string codes (A11, A12, etc.)
        # Includes Attribute19 (Telephone) and Attribute20 (foreign worker)
        # which have codes A191/A192 and A201/A202 respectively
        self.categorical_features = [
            'Attribute1', 'Attribute3', 'Attribute4', 'Attribute6',
            'Attribute7', 'Attribute9', 'Attribute10', 'Attribute12',
            'Attribute14', 'Attribute15', 'Attribute17', 'Attribute19', 'Attribute20'
        ]
        # Numerical (integer) features
        self.integer_features = [
            'Attribute2', 'Attribute5', 'Attribute8', 'Attribute11',
            'Attribute13', 'Attribute16', 'Attribute18'
        ]

    def fit(self, X, y):
        """
        Fit the pipeline on training data.

        :param X: Features DataFrame
        :param y: Target DataFrame/Series
        :return: self
        """
        X = X.copy()
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()

        self._fit_preprocessor(X)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        self.fitted = True

        return self

    def transform(self, X, y):
        """
        Transform features and target using fitted pipeline.

        :param X: Features DataFrame
        :param y: Target DataFrame/Series
        :return: Transformed (X, y) as numpy arrays
        """
        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        X = X.copy()
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()

        X_transformed = self.preprocessor.transform(X).astype(np.float32)
        y_transformed = self.label_encoder.transform(y)

        return X_transformed, y_transformed

    def fit_transform(self, X, y):
        """
        Fit and transform in one step (use for training data).

        :param X: Features DataFrame
        :param y: Target DataFrame/Series
        :return: Transformed (X, y) as numpy arrays
        """
        self.fit(X, y)
        return self.transform(X, y)

    def _fit_preprocessor(self, X):
        """Fit preprocessing: OneHotEncoder for categoricals, StandardScaler for integers."""
        transformers = [
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
             [col for col in self.categorical_features if col in X.columns]),
            ('num', StandardScaler(),
             [col for col in self.integer_features if col in X.columns])
        ]

        self.preprocessor = ColumnTransformer(
            transformers=[t for t in transformers if t[2]],
            remainder='drop'
        )
        self.preprocessor.fit(X)
