import numpy as np
import pandas as pd
import logging

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import itertools

from approach import helper
from sklearn.preprocessing import MinMaxScaler

from approach.abstract_predictor import PredictionModel


class LinearPredictor(PredictionModel):
    """
    LinearPredictor is a class that implements a regression model using ``SGDRegressor``

    This class extends the ``PredictionModel`` base class and provides methods for training, predicting and
    incrementally updating the model. It also includes functionality for selecting the best hyperparameters
    using cross validation.

    Attributes:
        X_train_full (np.ndarray): Historical training features.
        y_train_full (np.ndarray): Historical training labels.
        train_X_scaler (MinMaxScaler): Scaler for normalizing training features.
        train_y_scaler (MinMaxScaler): Scaler for normalizing training labels.
        regressor (SGDRegressor): Trained SGD regression model.
        model_error (float): Error score of the best model.
        err_metr (str): Error metric used for model selection.
    """

    params = {
        'alpha': [0.0001, 0.001],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'learning_rate': ['optimal', 'invscaling']
    }

    def __init__(self, workflow_name: str, task_name: str, err_metr: str,
                 batch_size: int = 1, retrain_interval: int | None = None,
                 use_online_grid: bool = False,
                 num_full_online_partials: int = 10,
                 save_top_n: int = 10,
                 param_cull_cutoff: int = -5,
                 param_cull_plus_percent: float = 0.3,
                 param_cull_minus_percent: float = 0.3):
        """Initialize the predictor.

        ``batch_size`` controls after how many samples the accumulated mini-
        batch is used for ``partial_fit``. ``retrain_interval`` defines after how
        many mini-batches a full re-training of the model should be triggered.
        When ``None`` or ``0`` no periodic re-training is performed.
        """
        super().__init__(workflow_name, task_name, err_metr)
        self.batch_size = batch_size
        self.retrain_interval = retrain_interval or 0
        self._update_counter = 0
        self._batch_X = []
        self._batch_y = []
        self.use_online_grid = use_online_grid
        if self.use_online_grid:
            self.param_order = tuple(self.params.keys())
            self.param_grid = {k: None for k in itertools.product(*self.params.values())}
            self.param_points = {k: 0 for k in itertools.product(*self.params.values())}
            self.num_full_online_partials = num_full_online_partials
            self.save_top_n = save_top_n
            self.param_cull_cutoff = param_cull_cutoff
            self.param_cull_plus_percent = param_cull_plus_percent
            self.param_cull_minus_percent = param_cull_minus_percent

    def initial_model_training(self, X_train, y_train) -> None:
        """
        Initializes the SGD regression model with training data.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        self.X_train_full = self._ensure_column_vector(X_train)
        self.y_train_full = self._ensure_column_vector(y_train)

        if self.use_online_grid:
            self._initial_online_grid(self.X_train_full, self.y_train_full)
        else:
            self._select_best_model(self.X_train_full, self.y_train_full)

    def predict_task(self, task_features: pd.Series) -> float:
        """
        Predicts the output for multiple tasks using the trained SGD regression model.

        :param task_features: Features of the task to predict.

        :return: Predicted value for the task.
        """
        task_features_scaled = self.train_X_scaler.transform(self._ensure_column_vector(task_features))
        preds = self.regressor.predict(task_features_scaled)
        return self.train_y_scaler.inverse_transform(self._ensure_column_vector(preds))

    def predict_tasks(self, taskDataframe: pd.DataFrame) -> np.ndarray:
        """
        Predicts the output for multiple tasks using the trained SGD regression model.

        :param taskDataframe: DataFrame containing features of multiple tasks.

        :return: Array of predicted values for the tasks.
        """
        taskDataframe_scaled = self.train_X_scaler.transform(taskDataframe)

        preds = self.regressor.predict(taskDataframe_scaled)
        return self.train_y_scaler.inverse_transform(self._ensure_column_vector(preds))

    def update_model(self, X_train: pd.Series, y_train: float) -> None:
        """Accumulate new samples and update the model in mini-batches."""
        if self.use_online_grid:
            self._update_online_grid(np.asarray(X_train), y_train)
            return
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

        # Update scalers with the mini-batch
        self.train_X_scaler.partial_fit(X_batch)
        self.train_y_scaler.partial_fit(y_batch)

        X_scaled = self.train_X_scaler.transform(X_batch)
        y_scaled = self.train_y_scaler.transform(y_batch).ravel()

        # Incrementally update the regressor on the whole mini-batch
        self.regressor.partial_fit(X_scaled, y_scaled)

        self._update_counter += 1
        if self.retrain_interval and self._update_counter % self.retrain_interval == 0:
            # Re-run full training to refresh weights similar to the initial grid-search
            self._select_best_model(self.X_train_full, self.y_train_full)

    def smoothed_mape(self, y_true, y_pred, epsilon=1e-8):
        """
        Calculates the smoothed Mean Absolute Percentage Error (MAPE) between true and predicted values.

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

    def _select_best_model(self, X_train, y_train):
        """
        Fit an ``SGDRegressor`` using grid search to select the best hyperparameters.

        This method is executed only once during initial training.

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
        }

        model = SGDRegressor(random_state=42, max_iter=1000)

        if self.err_metr == 'smoothed_mape':
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, error_score="raise", n_jobs=-1,
                                       scoring=smoothed_mape_scorer)
        elif self.err_metr == 'neg_mean_squared_error':
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, error_score="raise", n_jobs=-1,
                                       scoring='neg_mean_squared_error')
        else:
            raise NotImplementedError('Error metric not found.')
        grid_search.fit(X_train_scaled, y_train_scaled)

        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

        self.model_error = best_score
        self.regressor = best_model

    # ------------------------------------------------------------------
    # Online grid-search style training used when ``use_online_grid`` is True
    # ------------------------------------------------------------------

    def _initial_online_grid(self, X_train, y_train) -> None:
        self.train_X_scaler = MinMaxScaler()
        self.train_y_scaler = MinMaxScaler()

        X_scaled = self.train_X_scaler.fit_transform(X_train)
        y_scaled = self.train_y_scaler.fit_transform(self._ensure_column_vector(y_train))

        scores = []
        for params in self.param_grid.keys():
            if self.param_points[params] <= self.param_cull_cutoff:
                continue
            model = SGDRegressor(random_state=42, max_iter=1000,
                                 **{k: v for k, v in zip(self.param_order, params)})
            model.fit(X_scaled, y_scaled.ravel())
            self.param_grid[params] = model
            scores.append((params, model, model.score(X_scaled, y_scaled.ravel())))

        scores.sort(key=lambda x: x[2])

        for params, _, _ in scores[:min(int(round(len(scores) * self.param_cull_minus_percent)) + 1,
                                       max(0, len(scores) - self.save_top_n))]:
            self.param_points[params] -= 1

        for params, _, _ in scores[min(int(round(len(scores) * (1 - self.param_cull_plus_percent))) + 1,
                                       max(0, len(scores) - self.save_top_n)):]:
            self.param_points[params] += 1

        _, best_model, best_score = scores[-1]
        self.model_error = best_score
        self.regressor = best_model
        helper.log_if_verbose(f"Best Score for LinearRegressor: {best_score}")

    def _update_online_grid(self, X_train: np.ndarray, y_train: float) -> None:
        self.X_train_full = np.concatenate((self.X_train_full, [X_train]))
        self.y_train_full = np.concatenate((self.y_train_full, np.array([y_train]).reshape(-1, 1)))

        self.train_X_scaler = self.train_X_scaler.fit(self.X_train_full)
        self.train_y_scaler = self.train_y_scaler.fit(self.y_train_full)

        transformed_X = self.train_X_scaler.transform(self.X_train_full)
        transformed_y = self.train_y_scaler.transform(self.y_train_full).ravel()

        scores = []
        for params, model in self.param_grid.items():
            if self.param_points[params] <= self.param_cull_cutoff:
                continue
            model.partial_fit(self.train_X_scaler.transform(np.array(X_train).reshape(1, -1)),
                              self.train_y_scaler.transform(np.array([y_train]).reshape(1, -1)))
            early_quitting = [model.score(transformed_X, transformed_y)]
            for _ in range(self.num_full_online_partials):
                model.partial_fit(self.train_X_scaler.transform(self.X_train_full),
                                  self.train_y_scaler.transform(self.y_train_full).ravel())
                early_quitting.append(model.score(transformed_X, transformed_y))
                if len(early_quitting) >= 12:
                    if sum(helper.nth_deltas(early_quitting, 2, 10)) >= -1e-4:
                        break
            self.param_grid[params] = model
            scores.append((params, model, model.score(transformed_X, transformed_y)))

        scores.sort(key=lambda x: x[2])

        for params, _, _ in scores[:min(int(round(len(scores) * self.param_cull_minus_percent)) + 1,
                                       max(0, len(scores) - self.save_top_n))]:
            self.param_points[params] -= 1

        for params, _, _ in scores[min(int(round(len(scores) * (1 - self.param_cull_plus_percent))) + 1,
                                       max(0, len(scores) - self.save_top_n)):]:
            self.param_points[params] += 1

        _, best_model, best_score = scores[-1]
        self.model_error = best_score
        self.regressor = best_model
        helper.log_if_verbose(f"Best Score for LinearRegressor: {best_score}")

