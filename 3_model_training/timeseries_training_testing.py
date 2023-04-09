from sklearn.model_selection import TimeSeriesSplit
import pandas as pd


def timeseries_training_testing(df_final, model, split_count, gap=0, target_column='is_alarm',
                                date_column='date_tomorrow_epoch'):
    df_final['tomorrow_day'] = pd.to_datetime(df_final[date_column])
    # Initialize the time-series cross-validation object
    time_series = TimeSeriesSplit(n_splits=split_count, gap=gap)

    x = df_final.drop([target_column], axis=1)
    y = df_final[target_column]

    # Train and test the model on each fold of the cross-validation object
    for fold_i, (training_index, testing_index) in enumerate(time_series.split(x)):
        x_train, x_test = x.iloc[training_index], x.iloc[testing_index]
        y_train, y_test = y.iloc[training_index], y.iloc[testing_index]

        # Train the model on the training set
        model.fit(x_train, y_train)
        # Output testing of the model score
        print("Accuracy time series split:", fold_i, model.score(x_test, y_test))

    df_final[date_column] = df_final["tomorrow_day"].apply(
        lambda l: int(l.timestamp())
    )
    return model
