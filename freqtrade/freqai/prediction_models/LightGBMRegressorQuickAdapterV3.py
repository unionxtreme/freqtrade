import logging
from typing import Any, Dict, Tuple

from lightgbm import LGBMRegressor
import time
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import pandas as pd
import scipy as spy
import numpy.typing as npt
from pandas import DataFrame
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


class LightGBMRegressorQuickAdapterV3(BaseRegressionModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Most regressors use the same function names and arguments e.g. user
        can drop in LGBMRegressor in place of CatBoostRegressor and all data
        management will be properly handled by Freqai.
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = (data_dictionary["test_features"], data_dictionary["test_labels"])
            eval_weights = data_dictionary["test_weights"]
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        train_weights = data_dictionary["train_weights"]

        init_model = self.get_init_model(dk.pair)

        model = LGBMRegressor(**self.model_training_parameters)

        start = time.time()
        model.fit(X=X, y=y, eval_set=eval_set, sample_weight=train_weights,
                  eval_sample_weight=[eval_weights], init_model=init_model)
        time_spent = (time.time() - start)
        self.dd.update_metric_tracker('fit_time', time_spent, dk.pair)

        return model

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        filtered_df = dk.normalize_data_from_metadata(filtered_df)
        dk.data_dictionary["prediction_features"] = filtered_df

        # optional additional data cleaning/analysis
        self.data_cleaning_predict(dk)

        start = time.time()
        predictions = self.model.predict(dk.data_dictionary["prediction_features"])
        time_spent = (time.time() - start)
        self.dd.update_metric_tracker('predict_time', time_spent, dk.pair)

        pred_df = DataFrame(predictions, columns=dk.label_list)

        pred_df = dk.denormalize_labels_from_metadata(pred_df)

        return (pred_df, dk.do_predict)

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:

        warmed_up = True
        num_candles = self.freqai_info.get('fit_live_predictions_candles', 100)
        candle_diff = len(self.dd.historic_predictions[pair].index) - \
            (self.freqai_info.get('fit_live_predictions_candles', 600) + 1000)
        if candle_diff < 0:
            logger.warning(
                f'Fit live predictions not warmed up yet. Still {abs(candle_diff)} candles to go')
            warmed_up = False

        pred_df_full = self.dd.historic_predictions[pair].tail(num_candles).reset_index(drop=True)
        pred_df_sorted = pd.DataFrame()
        for label in pred_df_full.keys():
            if pred_df_full[label].dtype == object:
                continue
            pred_df_sorted[label] = pred_df_full[label]

        # pred_df_sorted = pred_df_sorted
        for col in pred_df_sorted:
            pred_df_sorted[col] = pred_df_sorted[col].sort_values(
                ascending=False, ignore_index=True)
        frequency = num_candles / (self.freqai_info['feature_parameters']['label_period_candles'] * 2)
        max_pred = pred_df_sorted.iloc[:int(frequency)].mean()
        min_pred = pred_df_sorted.iloc[-int(frequency):].mean()

        if not warmed_up:
            dk.data['extra_returns_per_train']['&s-maxima_sort_threshold'] = 2
            dk.data['extra_returns_per_train']['&s-minima_sort_threshold'] = -2
        else:
            dk.data['extra_returns_per_train']['&s-maxima_sort_threshold'] = max_pred['&s-extrema']
            dk.data['extra_returns_per_train']['&s-minima_sort_threshold'] = min_pred['&s-extrema']

        dk.data["labels_mean"], dk.data["labels_std"] = {}, {}
        for ft in dk.label_list:
            # f = spy.stats.norm.fit(pred_df_full[ft])
            dk.data['labels_std'][ft] = 0  # f[1]
            dk.data['labels_mean'][ft] = 0  # f[0]

        # fit the DI_threshold
        if not warmed_up:
            f = [0, 0, 0]
            cutoff = 2
        else:
            f = spy.stats.weibull_min.fit(pred_df_full['DI_values'])
            cutoff = spy.stats.weibull_min.ppf(0.999, *f)

        dk.data["DI_value_mean"] = pred_df_full['DI_values'].mean()
        dk.data["DI_value_std"] = pred_df_full['DI_values'].std()
        dk.data['extra_returns_per_train']['DI_value_param1'] = f[0]
        dk.data['extra_returns_per_train']['DI_value_param2'] = f[1]
        dk.data['extra_returns_per_train']['DI_value_param3'] = f[2]
        dk.data['extra_returns_per_train']['DI_cutoff'] = cutoff
