import os

from paths_rel import *

projectPath = os.path.join(os.environ["VIRTUAL_ENV"], "..")


ALARMS_DATA_FILE = os.path.join(projectPath, REL_ALARMS_DATA_FILE)
REGIONS_DATA_FILE = os.path.join(projectPath, REL_REGIONS_DATA_FILE)
ISW_TF_IDF_RESULT = os.path.join(projectPath, REL_ISW_TF_IDF_RESULT)
WEATHER_DATA_FILE = os.path.join(projectPath, REL_WEATHER_DATA_FILE)


HOLIDAYS_DATA_FOLDER = os.path.join(projectPath, REL_HOLIDAYS_DATA_FOLDER)
ISW_SCRAPPING_FOLDER = os.path.join(projectPath, REL_ISW_SCRAPPING_FOLDER)


RUS_HOLIDAYS_FILENAME_F = os.path.join(
    projectPath, HOLIDAYS_DATA_FOLDER + "/russian_holidays_{}.csv"
)
UKRAINIAN_HOLIDAYS_FILENAME_F = os.path.join(
    projectPath, HOLIDAYS_DATA_FOLDER + "/ukrainian_holidays_{}.csv"
)
