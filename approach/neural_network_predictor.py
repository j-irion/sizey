import numpy as np
import pandas as pd
import logging

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, LeaveOneOut, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning


from approach.abstract_predictor import PredictionModel

simplefilter("ignore", category=ConvergenceWarning)


class NeuralNetworkPredictor(PredictionModel):
    """
    NeuralNetworkPredictor is a class that implements a machine learning model for regression tasks using a Neural Network.

    This class extends the `PredictionModel` base class and provides methods for training, predicting, and updating
    the Neural Network model. It also includes functionality for hyperparameter tuning and model selection based on
    specified error metrics.

    Attributes:
        X_train_full (np.ndarray): Historical training features.
        y_train_full (np.ndarray): Historical training labels.
        train_X_scaler (MinMaxScaler): Scaler for normalizing training features.
        train_y_scaler (MinMaxScaler): Scaler for normalizing training labels.
        regressor (MLPRegressor): Trained Neural Network model.
        model_error (float): Error score of the best model.
        best_params (dict): Hyperparameters of the selected model.
        err_metr (str): Error metric used for model selection.
    """

    def __init__(self, workflow_name: str, task_name: str, err_metr: str,
                 batch_size: int = 1, retrain_interval: int | None = None):
        """Initialize the predictor.

        ``batch_size`` defines how many samples are accumulated before an
        incremental update is performed. ``retrain_interval`` controls after how
        many of these updates a full re-training of the model should be
        executed. Set it to ``None`` or ``0`` to disable periodic re-training.
        """
        super().__init__(workflow_name, task_name, err_metr)
        self.batch_size = batch_size
        self.retrain_interval = retrain_interval or 0
        self._update_counter = 0
        self._batch_X = []
        self._batch_y = []

    def initial_model_training(self, X_train, y_train) -> None:
        """
        Initializes the Neural Network model with training data.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        # Initialize internal storage of historical values
        self.X_train_full = self._ensure_column_vector(X_train)
        self.y_train_full = self._ensure_column_vector(y_train)

        self._selectBestModel(self.X_train_full, self.y_train_full)

    def predict_task(self, task_features: pd.Series) -> float:
        """
        Predicts the output for a single task based on its features using the trained Neural Network model.

        :param task_features: Features of the task to predict.

        :return: Predicted value for the task.
        """
        task_features_scaled = self.train_X_scaler.transform(self._ensure_column_vector(task_features))
        preds = self.regressor.predict(task_features_scaled)
        return self.train_y_scaler.inverse_transform(self._ensure_column_vector(preds))

    def predict_tasks(self, taskDataframe: pd.DataFrame) -> np.ndarray:
        """
        Predicts the output for multiple tasks based on their features using the trained Neural Network model.

        :param taskDataframe: DataFrame containing features of multiple tasks.

        :return: Array of predicted values for the tasks.
        """
        taskDataframe_scaled = self.train_X_scaler.transform(taskDataframe)

        preds = self.regressor.predict(taskDataframe_scaled)
        return self.train_y_scaler.inverse_transform(self._ensure_column_vector(preds))

    def update_model(self, X_train: pd.Series, y_train: float) -> None:
        """Accumulate new samples and update the model in mini-batches."""
        X_col = self._ensure_column_vector(X_train)
        y_col = self._ensure_column_vector([y_train])

        self._batch_X.append(X_col)
        self._batch_y.append(y_col)

        if len(self._batch_X) < self.batch_size:
            return

        X_batch = np.vstack(self._batch_X)
        y_batch = np.vstack(self._batch_y)
        self._batch_X.clear()
        self._batch_y.clear()

        # Keep history for evaluation only
        self.X_train_full = np.concatenate((self.X_train_full, X_batch))
        self.y_train_full = np.concatenate((self.y_train_full, y_batch))

        # Update scalers with new data
        # Update scalers with the mini-batch
        self.train_X_scaler.partial_fit(X_batch)
        self.train_y_scaler.partial_fit(y_batch)

        X_scaled = self.train_X_scaler.transform(X_batch)
        y_scaled = self.train_y_scaler.transform(y_batch).ravel()

        # Incrementally update the regressor
        self.regressor.partial_fit(X_scaled, y_scaled)

        self._update_counter += 1
        if self.retrain_interval and self._update_counter % self.retrain_interval == 0:
            # Refit model on the accumulated dataset to refresh weights
            self._selectBestModel(self.X_train_full, self.y_train_full)

    def smoothed_mape(self, y_true, y_pred, epsilon=1e-8):
        """
        Computes the smoothed Mean Absolute Percentage Error (MAPE) between true and predicted values.

        :param y_true: True values.
        :param y_pred: Predicted values.
        :param epsilon: Small value to avoid division by zero.

        :return: Smoothed MAPE value.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Calculate the individual percentage errors and clip each at 100%
        mape = np.abs((y_true - y_pred) / (y_true + epsilon))
        mape = -1 * np.clip(mape, None, 1)  # Clip values at 100%
        return np.mean(mape)

    def _selectBestModel(self, X_train, y_train):
        """
        Fit an ``MLPRegressor`` using grid search to select the best hyperparameters.

        This method is executed only once during the initial training. The best
        parameters are stored so that subsequent calls to ``update_model`` can
        continue training the same model with ``partial_fit`` without running
        grid search again.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        self.train_X_scaler = MinMaxScaler()
        self.train_y_scaler = MinMaxScaler()

        # Scale Features
        X_train_scaled = self.train_X_scaler.fit_transform(X_train)
        y_train_scaled = self.train_y_scaler.fit_transform(self._ensure_column_vector(y_train))

        smoothed_mape_scorer = make_scorer(self.smoothed_mape, greater_is_better=True)

        param_grid = {
            'hidden_layer_sizes': [(5, 5), (7, 7), (9, 9)],
            'alpha': [0.0001, 0.001],
            'solver': ['lbfgs', 'adam'],
            'learning_rate': ['constant', 'adaptive']
        }

        model = MLPRegressor(random_state=42, max_iter=5000)
        if self.err_metr == 'smoothed_mape':
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, error_score="raise", n_jobs=-1,
                                       scoring=smoothed_mape_scorer)
        elif self.err_metr == 'neg_mean_squared_error':
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, error_score="raise", n_jobs=-1,
                                       scoring='neg_mean_squared_error')
        else:
            raise NotImplementedError('Error metric not found.')
        grid_search.fit(X_train_scaled, y_train_scaled.ravel())

        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

        self.model_error = best_score
        self.best_params = grid_search.best_params_
        self.regressor = best_model


