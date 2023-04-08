# import numpy as np

# from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LinearRegression

# df_final = df_weather_v6

# # Fill NaN values
# df_final[['event_all_region']] = df_final[['event_all_region']].fillna(value=0)

# # Separating the data into independent and dependent variables
# # Converting each dataframe into a numpy array
# X = np.array(df_final[['region_id', 'event_all_region', 'day_datetimeEpoch', 'hour_datetimeEpoch', 'ukrainian_holiday', 'russian_holiday', 'hour_temp', 'hour_snow', 'hour_visibility', 'hour_conditions_code', 'lunar_eclipse', 'solar_eclipse', 'moonphased_eclipse','alarmed_regions_count']])
# y = np.array(df_final['is_alarm'])

# # Dropping any rows with Nan values
# # df_final.dropna(inplace = True)

# # Splitting the data into training and testing data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

# # LinerRegression
# regr = LinearRegression()
# #
# regr.fit(X_train, y_train)
# #
# print(regr.score(X_test, y_test))
# # df_weather_v4 -      0.7357812097367479
# # df_weather_v5 -      0.728836911878402
# # hours with no alarm - 0.7475497734309323