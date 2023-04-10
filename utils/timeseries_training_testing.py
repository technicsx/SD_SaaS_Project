from sklearn.model_selection import TimeSeriesSplit

def timeseries_training_testing(df_final, model, split_count, gap=0, target_column='is_alarm'):
    # Initialize the time-series cross-validation object
    time_series = TimeSeriesSplit(n_splits=split_count, gap=gap)

    x = df_final.drop([target_column], axis=1)
    y = df_final[target_column]

    # Train and test the model on each fold of the cross-validation object
    for training_index, testing_index in time_series.split(x):
        x_train, x_test = x.iloc[training_index], x.iloc[testing_index]
        y_train, y_test = y.iloc[training_index], y.iloc[testing_index]

        # Train the model on the training set
        model.fit(x_train, y_train)
        # Output testing of the model score
        print("Accuracy time series split:",  model.score(x_test, y_test))

    return model
