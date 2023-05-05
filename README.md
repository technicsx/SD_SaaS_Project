## Team members:
Kuchmenko Yaroslav, Kopiika Vadym, Nakytniak Vadym, Polishchuk Yurii

# Air Raid Prediction SaaS service for Ukrainan regions during russian War against Ukraine

This is a part of the project which consists of crucial logic for prediction model training and testing.

Feel free to checkout other project parts:
- <https://github.com/lemonderon/Air_Raid_Prediction_SaaS> for backend part of the project.
- <https://github.com/lemonderon/SD_SaaS_Project_App> for simple Flutter cross-platform frontend application.

## Quick summary
- Project consists of foldered structure of lua and python code for perfroming such steps:
  - 0 - data scrapping
  - 1 - data preparation (of scrapped data)
  - 2 - data visualization (of prepared data)
  - 3 - models training for TOP-3 picked (based on prepared data)
  
It also has supporting materials such as external_data contents, features folder with methods for feature creation and utils folder that consists of some handy methods.

### Data scrapping: consists of code for scrapping ISW reports, forecasted/historical weather and air raid alarms history for Ukrainian regions and code for scrapping logs from air_alert telegram bot.

### Data preparation: consists of code for preparing ISW tf-idf based on preprocessed reports, merging logic for getting finalized data for model training, model prediction and model retraining with supporting code such as formating of telegram logs scrapped from telegram bot. (It is also the place where merged dataset is having normalization and encoding for specific columns)

### Data visualization: consists of visualization logic for alarms, ISWs, weather and merged dataset.

### Models training: consists of training logic for 3 models (random forest, gradient boosting and logistic regression) with supporting code for hyperparameters tuning based on prepared data.

## In order to get trained models metioned above:
- Setup python interpreter using poetry, see docs: https://python-poetry.org/docs/
- After that run poetry install to install all dependencies.
- Finally, you will be able to follow mentioned folder structure with numbered naming of them, starting from 0 and ending with 3.
- Everything can be run as python files. If needed there is possibility to use "jupytext --update --to notebook . file_path.py" command to receive easy to read notebooks with .ipynb extension and run them.
