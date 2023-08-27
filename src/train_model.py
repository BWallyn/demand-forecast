"""Train model using mlflow"""
# =================
# ==== IMPORTS ====
# =================

import json
import os
import numpy as np
import pandas as pd

from typing import List, Optional

from catboost import Pool, CatBoostRegressor
import shap
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.catboost

import matplotlib.pyplot as plt


# ===================
# ==== FUNCTIONS ====
# ===================

def unsold_rate(delivered: pd.Series, sales: pd.Series) -> float:
        """Measures the proportion of the number of unsold publication compared 
        to the number of delivered publication.

        Args:
            delivered: List of delivered publications.
            sales: List of sold publications.

        Returns:
            float: The unsold rate.
        """
        return (delivered.sum() - sales.sum()) / delivered.sum()


def sold_out_rate(delivered: pd.Series, sales: pd.Series) -> float:
        """Measures the frequency of sold out cases.
        
        Args:
            delivered: List of delivered publications.
            sales: List of sold publications.

        Returns:
            float: The sold out rate.
        """
        return (delivered == sales).sum() / len(delivered)


def mlflow_log_parameters(model: CatBoostRegressor) -> None:
    """Log the parameters of the Catboost regressor model to MLflow

    Args:
        model: Catboost regressor model trained
    """
    all_params = model.get_all_params()
    mlflow.log_param('depth', all_params['depth'])
    mlflow.log_param('iterations', all_params['iterations'])
    mlflow.log_param('loss_function', all_params['loss_function'])
    mlflow.log_param('learning_rate', all_params['learning_rate'])
    mlflow.log_param('l2_leaf_reg', all_params['l2_leaf_reg'])
    mlflow.log_param('random_strength', all_params['random_strength'])
    mlflow.log_param('border_count', all_params['border_count'])


def mlflow_log_model(model: CatBoostRegressor) -> None:
    """Log the Catboost regressor model to MLflow

    Args:
        model: Catboost regressor model trained
    """
    # model.save_model('../models/cb_classif')
    mlflow.catboost.log_model(cb_model=model, artifact_path='model')


def mlflow_log_metrics(deliv_train: np.array, deliv_eval: np.array, sales_train: np.array, sales_eval: np.array) -> None:
    """Log metrics to MLflow

    Args:
        deliv_train: prediction on the train set
        deliv_eval: prediction on the evaluation set
        sales_train: target of the train set
        sales_eval: target of the evaluation set
    """
    mlflow.log_metric('unsold_train', unsold_rate(deliv_train, sales_train))
    mlflow.log_metric('unsold_eval', unsold_rate(deliv_eval, sales_eval))
    mlflow.log_metric('soldout_train', sold_out_rate(deliv_train, sales_train))
    mlflow.log_metric('soldout_eval', sold_out_rate(deliv_eval, sales_eval))
    mlflow.log_metric('rmse_train', mean_squared_error(sales_train, deliv_train, squared=True))
    mlflow.log_metric('rmse_eval', mean_squared_error(sales_eval, deliv_eval, squared=True))


def mlflow_log_shap(model: CatBoostRegressor, df_train: pd.DataFrame, shap_max_disp: int, path_reports: str) -> None:
    """Log the shap values in an aritfact to MLflow

    Args:
        model: Catboost regressor model trained
        df_train: training set
        shap_max_display: top features to display in the shap plot
        path_reports: path to the reports folder to save figures
    """
    path_fig = os.path.join(path_reports, 'shap_beeswarm.png')
    # shap.initjs()
    explainer = shap.Explainer(model)
    shap_values = explainer(df_train)
    shap.plots.beeswarm(shap_values, max_display=shap_max_disp, show=False)
    plt.savefig(path_fig)
    plt.show()
    mlflow.log_artifact(path_fig, artifact_path="feat")


def train_model_mlflow(
    df_train: pd.DataFrame, df_eval: pd.DataFrame, y_train: pd.DataFrame, y_eval: pd.DataFrame,
    feat_cat: List[str], plot_training: Optional[bool]=False, verbose: Optional[int]=0,
    shap_max_disp: Optional[int]=10, path_reports: Optional[str]='../reports',
    **kwargs
) -> tuple[CatBoostRegressor, np.array, np.array]:
    """Train a Catboost regressor model and log the parameters, metrics, model and shap values to MLflow

    Create the catboost pools for the catboost model
    Create a MLflow run
    Define a Catboost regressor model
    Train the model on training set and use a validation set to keep the best model
    Predict on train and validation sets
    Log parameters, model, metrics, shap and confusion matrix to MLflow

    Args:
        df_train: training set
        df_eval: validation set
        y_train: target of the training set
        y_eval: target of the validation set
        feat_cat: list of categorical features
        plot_training: whether to plot the leaning curves
        verbose: verbose parameter while learning
        shap_max_display: top features to display on the shap values
        **kwargs: hyperparameters of the Catboost regressor
    Returns:
        model: Catboost regressor model trained
        pred_train: predictions on the train set
        pred_valid: predictions on the validation set
    """
    # Create Pools for catboost model
    pool_train = Pool(df_train, label=y_train.values, cat_features=feat_cat)
    pool_eval = Pool(df_eval, label=y_eval.values, cat_features=feat_cat)
    # MLflow
    with mlflow.start_run():
        model = CatBoostRegressor(
            random_seed=12,
            **kwargs
        )
        model.fit(
            pool_train,
            eval_set=pool_eval,
            use_best_model=True,
            plot=plot_training,
            verbose=verbose,
        )
        # Predict
        pred_train = model.predict(pool_train)
        pred_eval = model.predict(pool_eval)
        # Log parameters to mlflow
        mlflow_log_parameters(model)
        mlflow_log_model(model)
        mlflow_log_metrics(pred_train, pred_eval, y_train.values, y_eval.values)
        mlflow_log_shap(model, df_train, shap_max_disp=shap_max_disp, path_reports=path_reports)
    return model, pred_train, pred_eval


def mlflow_log_metrics_cv(
    l_unsold_rate_train: list[float], l_unsold_rate_eval: list[float],
    l_soldout_rate_train: list[float], l_soldout_rate_eval: list[float],
    l_rmse_train: list[float], l_rmse_eval: list[float], best_iteration: int
) -> None:
    """Log metrics to MLflow

    Args:
        l_unsold_rate_train: list of the unsold rates for the train set
        l_unsold_rate_eval: list of the unsold rates for the evaluation set
        l_soldout_rate_train: list of the sold out rates for the train set
        l_soldout_rate_eval: list of the sold out rates for the evaluation set
        l_rmse_train: list of the RMSE for the train set
        l_rmse_eval: list of the RMSE for the evaluation set
    """
    mlflow.log_metric('unsold_mean_train', np.mean(l_unsold_rate_train))
    mlflow.log_metric('unsold_mean_eval', np.mean(l_unsold_rate_eval))
    mlflow.log_metric('soldout_mean_train', np.mean(l_soldout_rate_train))
    mlflow.log_metric('soldout_mean_eval', np.mean(l_soldout_rate_eval))
    mlflow.log_metric('rmse_mean_train', np.mean(l_rmse_train))
    mlflow.log_metric('rmse_mean_eval', np.mean(l_rmse_eval))
    mlflow.log_metric('best_iteration', best_iteration)


def train_model_cv_mlflow(
    list_train_valid: list[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
    feat_cat: List[str], plot_training: Optional[bool]=False, verbose: Optional[int]=0,
    shap_max_disp: Optional[int]=10, path_reports: Optional[str]='../reports',
    **kwargs
) -> tuple[CatBoostRegressor, np.array, np.array]:
    """Using cross validation, train a Catboost regressor model and log the parameters, metrics, model and shap values to MLflow

    Create a MLflow run
    For each split of train and validation, using expanding window validation:
    - Create the catboost pools for the catboost model
    - Define a Catboost regressor model
    - Train the model on training set and use a validation set to keep the best model
    - Predict on train and evaluation sets
    Log parameters, model, metrics, shap and confusion matrix to MLflow

    Args:
        list_train_valid: list of the different splits of train and validation
        feat_cat: list of categorical features
        plot_training: whether to plot the leaning curves
        verbose: verbose parameter while learning
        shap_max_display: top features to display on the shap values
        **kwargs: hyperparameters of the Catboost regressor
    Returns:
        model: Catboost regressor model trained
        pred_train: predictions on the train set
        pred_valid: predictions on the validation set
    """
    # Options
    l_unsold_rate_train = []
    l_unsold_rate_eval = []
    l_soldout_rate_train = []
    l_soldout_rate_eval = []
    l_rmse_train = []
    l_rmse_eval = []
    # MLflow
    with mlflow.start_run():
        # Run over the different folds
        for i in range(len(list_train_valid)):
            df_train_, df_eval_, y_train_, y_eval_ = list_train_valid[i]
            # Create Pools for catboost model
            pool_train = Pool(df_train_, label=y_train_.values, cat_features=feat_cat)
            pool_eval = Pool(df_eval_, label=y_eval_.values, cat_features=feat_cat)
            # Create model
            model = CatBoostRegressor(
                random_seed=12,
                **kwargs
            )
            # Fit model
            model.fit(
                pool_train,
                eval_set=pool_eval,
                use_best_model=True,
                plot=plot_training,
                verbose=verbose,
            )
            # Predict
            pred_train = model.predict(pool_train)
            pred_eval = model.predict(pool_eval)
            l_unsold_rate_train.append(unsold_rate(pred_train, y_train_.values))
            l_unsold_rate_eval.append(unsold_rate(pred_eval, y_eval_.values))
            l_soldout_rate_train.append(sold_out_rate(pred_train, y_train_.values))
            l_soldout_rate_eval.append(sold_out_rate(pred_eval, y_eval_.values))
            l_rmse_train.append(mean_squared_error(y_train_.values, pred_train, squared=True))
            l_rmse_eval.append(mean_squared_error(y_eval_.values, pred_eval, squared=True))
        # Log parameters to mlflow
        mlflow_log_metrics_cv(
            l_unsold_rate_train, l_unsold_rate_eval, l_soldout_rate_train, l_soldout_rate_eval, l_rmse_train, l_rmse_eval,
            model.get_best_iteration()
        )
        mlflow_log_shap(model, df_train_, shap_max_disp=shap_max_disp, path_reports=path_reports)
        mlflow_log_model(model)
        mlflow_log_parameters(model)
        return model, pred_train, pred_eval
