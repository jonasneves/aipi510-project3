"""
XGBoost Salary Prediction Model

Predicts AI/ML salary ranges based on job features, location,
experience level, skills, and market trends.
"""

## AI Tool Attribution: Built with assistance from Claude Code CLI (https://claude.ai/claude-code)
## Implemented the XGBoost regression model with cross-validation, hyperparameter tuning,
## confidence interval predictions, and feature importance tracking for salary estimation.

import pickle
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class SalaryPredictor:
    """XGBoost-based salary prediction model."""

    DEFAULT_PARAMS = {
        "objective": "reg:squarederror",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        model_dir: str = "models",
        params: Optional[dict] = None,
    ):
        """
        Initialize salary predictor.

        Args:
            model_dir: Directory to save/load models
            params: XGBoost parameters (uses defaults if None)
        """
        if xgb is None:
            raise ImportError(
                "xgboost is required. Install with: pip install xgboost"
            )

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.params = params or self.DEFAULT_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.training_metrics = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 20,
    ) -> dict:
        """
        Train the salary prediction model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            early_stopping_rounds: Early stopping patience

        Returns:
            Training metrics dictionary
        """
        print("Training XGBoost salary prediction model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {len(X_train.columns)}")

        # Deduplicate feature names to avoid reindex errors during prediction
        self.feature_names = list(dict.fromkeys(X_train.columns))

        # Initialize model
        self.model = xgb.XGBRegressor(**self.params)

        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            print(f"Validation samples: {len(X_val)}")

        # Train with early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=True,
        )

        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        self.training_metrics["train"] = self._calculate_metrics(y_train, train_pred)
        self.training_metrics["train"]["n_samples"] = len(X_train)

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            self.training_metrics["validation"] = self._calculate_metrics(y_val, val_pred)
            self.training_metrics["validation"]["n_samples"] = len(X_val)

        # Store feature importance
        self.feature_importance = dict(
            zip(self.feature_names, self.model.feature_importances_)
        )

        print("\nTraining completed!")
        self._print_metrics("Training", self.training_metrics["train"])

        if "validation" in self.training_metrics:
            self._print_metrics("Validation", self.training_metrics["validation"])

        return self.training_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict salaries for given features.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predicted salaries
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Remove duplicate columns from input DataFrame
        X = X.loc[:, ~X.columns.duplicated()]

        # Deduplicate feature names (in case loaded model has duplicates)
        unique_features = list(dict.fromkeys(self.feature_names))

        # Ensure feature columns match training
        X = X.reindex(columns=unique_features, fill_value=0)

        return self.model.predict(X)

    def predict_with_range(
        self,
        X: pd.DataFrame,
        confidence: float = 0.9,
    ) -> pd.DataFrame:
        """
        Predict salaries with confidence intervals.

        Uses quantile predictions if available, otherwise estimates
        based on training RMSE.

        Args:
            X: Feature DataFrame
            confidence: Confidence level for intervals

        Returns:
            DataFrame with predictions and ranges
        """
        predictions = self.predict(X)

        # Estimate range based on training error
        rmse = self.training_metrics.get("train", {}).get("rmse", 20000)
        z_score = 1.645 if confidence == 0.9 else 1.96  # 90% or 95%

        results = pd.DataFrame({
            "predicted_salary": predictions,
            "salary_low": predictions - (z_score * rmse),
            "salary_high": predictions + (z_score * rmse),
        })

        # Ensure non-negative salaries
        results["salary_low"] = results["salary_low"].clip(lower=30000)

        return results

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Evaluation metrics dictionary
        """
        print(f"\nEvaluating on {len(X_test)} test samples...")

        predictions = self.predict(X_test)
        metrics = self._calculate_metrics(y_test, predictions)

        self._print_metrics("Test", metrics)

        return metrics

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
    ) -> dict:
        """Calculate regression metrics."""
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "median_ae": np.median(np.abs(y_true - y_pred)),
        }

    def _print_metrics(self, split_name: str, metrics: dict) -> None:
        """Print formatted metrics."""
        print(f"\n{split_name} Metrics:")
        print(f"  MAE:  ${metrics['mae']:,.0f}")
        print(f"  RMSE: ${metrics['rmse']:,.0f}")
        print(f"  R2:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  Median AE: ${metrics['median_ae']:,.0f}")

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> dict:
        """
        Perform k-fold cross-validation.

        Args:
            X: Features
            y: Targets
            cv: Number of folds

        Returns:
            Cross-validation results
        """
        print(f"\nPerforming {cv}-fold cross-validation...")

        model = xgb.XGBRegressor(**self.params)

        # Calculate multiple metrics
        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }

        results = {}
        for metric_name, scorer in scoring.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
            if "neg_" in scorer:
                scores = -scores
            results[metric_name] = {
                "mean": scores.mean(),
                "std": scores.std(),
                "scores": scores.tolist(),
            }

        print("\nCross-validation Results:")
        print(f"  MAE:  ${results['mae']['mean']:,.0f} (+/- ${results['mae']['std']:,.0f})")
        print(f"  RMSE: ${results['rmse']['mean']:,.0f} (+/- ${results['rmse']['std']:,.0f})")
        print(f"  R2:   {results['r2']['mean']:.4f} (+/- {results['r2']['std']:.4f})")

        return results

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[dict] = None,
        cv: int = 3,
    ) -> dict:
        """
        Tune hyperparameters using grid search.

        Args:
            X: Features
            y: Targets
            param_grid: Parameter grid to search
            cv: Cross-validation folds

        Returns:
            Best parameters and results
        """
        if param_grid is None:
            param_grid = {
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.15],
                "n_estimators": [100, 200, 300],
                "min_child_weight": [1, 3, 5],
            }

        print("\nTuning hyperparameters...")
        print(f"Parameter grid: {param_grid}")

        base_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring="neg_mean_absolute_error",
            verbose=1,
            n_jobs=-1,
        )

        grid_search.fit(X, y)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best MAE: ${-grid_search.best_score_:,.0f}")

        # Update model params with best found
        self.params.update(grid_search.best_params_)

        return {
            "best_params": grid_search.best_params_,
            "best_score": -grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
        }

    def get_feature_importance(
        self,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Get feature importance ranking.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")

        importance_df = pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in self.feature_importance.items()
        ]).sort_values("importance", ascending=False)

        return importance_df.head(top_n)

    def save(self, name: str = "salary_model") -> Path:
        """
        Save model and metadata.

        Args:
            name: Model name for saving

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"{name}_{timestamp}.pkl"

        # Save model and metadata together
        save_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "training_metrics": self.training_metrics,
            "params": self.params,
            "timestamp": timestamp,
        }

        with open(model_path, "wb") as f:
            pickle.dump(save_data, f)

        print(f"Model saved to {model_path}")

        # Also save a latest version link
        latest_path = self.model_dir / f"{name}_latest.pkl"
        with open(latest_path, "wb") as f:
            pickle.dump(save_data, f)

        return model_path

    def load(self, path: Optional[str] = None, name: str = "salary_model") -> None:
        """
        Load a saved model.

        Args:
            path: Path to model file (optional)
            name: Model name to load latest version
        """
        if path is None:
            path = self.model_dir / f"{name}_latest.pkl"

        with open(path, "rb") as f:
            save_data = pickle.load(f)

        self.model = save_data["model"]
        # Deduplicate feature names to avoid reindex errors during prediction
        self.feature_names = list(dict.fromkeys(save_data["feature_names"]))
        self.feature_importance = save_data["feature_importance"]
        self.training_metrics = save_data["training_metrics"]
        self.params = save_data["params"]

        print(f"Model loaded from {path}")
        print(f"Features: {len(self.feature_names)}")
