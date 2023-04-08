import os

from paths_rel import *

projectPath = os.path.join(os.environ["VIRTUAL_ENV"], "..")


ALARMS_DATA_FILE = os.path.join(projectPath, REL_ALARMS_DATA_FILE)
REGIONS_DATA_FILE = os.path.join(projectPath, REL_REGIONS_DATA_FILE)
ISW_TF_IDF_RESULT = os.path.join(projectPath, REL_ISW_TF_IDF_RESULT)
WEATHER_DATA_FILE = os.path.join(projectPath, REL_WEATHER_DATA_FILE)


HOLIDAYS_DATA_FOLDER = os.path.join(projectPath, REL_HOLIDAYS_DATA_FOLDER)
ECLIPSES_DATA_FOLDER = os.path.join(projectPath, REL_ECLIPSES_DATA_FOLDER)
ISW_SCRAPPING_FOLDER = os.path.join(projectPath, REL_ISW_SCRAPPING_FOLDER)
print(HOLIDAYS_DATA_FOLDER)
print(HOLIDAYS_DATA_FOLDER)
print(HOLIDAYS_DATA_FOLDER)
print(HOLIDAYS_DATA_FOLDER)

RUS_HOLIDAYS_FILENAME_F = os.path.join(HOLIDAYS_DATA_FOLDER, "russian_holidays_{}.csv")
UKRAINIAN_HOLIDAYS_FILENAME_F = os.path.join(
    HOLIDAYS_DATA_FOLDER, "ukrainian_holidays_{}.csv"
)
LUNAR_ECLIPSES_FILENAME_F = os.path.join(
    ECLIPSES_DATA_FOLDER, "lunar_eclipses_flow_{}.csv"
)
SOLAR_ECLIPSES_FILENAME_F = os.path.join(
    ECLIPSES_DATA_FOLDER, "solar_eclipses_flow_{}.csv"
)
