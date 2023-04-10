import pandas as pd

from paths_full import UKRAINIAN_HOLIDAYS_FILENAME_F, RUS_HOLIDAYS_FILENAME_F


def add_ukrainian_holidays(
    dataset, day_datetime_column="day_datetime", column_name="ukrainian_holiday"
):
    add_holidays(
        dataset,
        day_datetime_column,
        column_name,
        UKRAINIAN_HOLIDAYS_FILENAME_F,
        [2022, 2023],
    )


def add_russian_holidays(
    dataset, day_datetime_column="day_datetime", column_name="russian_holiday"
):
    add_holidays(
        dataset, day_datetime_column, column_name, RUS_HOLIDAYS_FILENAME_F, [2022, 2023]
    )


def add_holidays(dataset, day_datetime_column, column_name, holidays_filename, years):
    holidays = []
    for year in years:
        filename = holidays_filename.format(year)
        holidays_df = pd.read_csv(filename)
        holidays_df["date"] = pd.to_datetime(holidays_df["date"])
        holidays.extend(holidays_df["date"].values)
    dataset[column_name] = dataset[day_datetime_column].apply(
        lambda x: 1 if x in holidays else 0
    )
