import pandas as pd

from paths_full import LUNAR_ECLIPSES_FILENAME_F, SOLAR_ECLIPSES_FILENAME_F


def add_lunar_eclipses(
    dataset,
    day_datetime_column="day_datetime",
    column_name_1="lunar_eclipse",
    column_name_2="moonphased_eclipse",
):
    add_astrology(
        dataset,
        day_datetime_column,
        column_name_1,
        column_name_2,
        '../features/eclipses_feature/csv/lunar_eclipses_flow.csv',
        [2022, 2023],
    )


def add_solar_eclipses(
    dataset,
    day_datetime_column="day_datetime",
    column_name_1="solar_eclipse",
    column_name_2="moonphased_eclipse",
):
    add_astrology(
        dataset,
        day_datetime_column,
        column_name_1,
        column_name_2,
        '../features/eclipses_feature/csv/lunar_eclipses_flow.csv',
        [2022, 2023],
    )


def add_astrology(
    dataset, day_datetime_column, column_name_1, column_name_2, eclipses_filename, years
):
    eclipses = []

    for year in years:
        filename = eclipses_filename.format(year)
        eclipses_df = pd.read_csv(filename)
        eclipses_df["date"] = pd.to_datetime(eclipses_df["date"])
        eclipses.extend(eclipses_df["date"].values)

    dataset[column_name_1] = dataset[day_datetime_column].apply(
        lambda x: 1 if x in eclipses else 0
    )
    dataset[column_name_2] = dataset.apply(
        lambda x: 1
        if x[day_datetime_column] in eclipses and float(x["day_moonphase"]) >= 0.90
        else 0,
        axis=1,
    )
