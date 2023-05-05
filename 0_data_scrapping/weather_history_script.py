import datetime as dt
import requests
import os
import csv
import datetime
import os
from dotenv import load_dotenv
import pandas as pd
import json

load_dotenv()

CROSS_VIS_WEATHER_API_KEY = os.environ.get("CROSS_VIS_WEATHER_API_KEY")

DT_FORMATTING = "%Y-%m-%dT%H:%M:%S"
DEFAULT_HOURS = 12


def format_history_from_weather_api(
):
    directory = './scripts/0_data_scrapping/results'
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath) as f:
                data = json.load(f)
                # Load the JSON data

                # Extract the required fields
                city_address = data['address']
                city_timezone = data['timezone']
                city_latitude = data['latitude']
                city_longitude = data['longitude']
                city_resolvedAddress = data['resolvedAddress']
                city_address = data['address']
                city_timezone = data['timezone']
                city_tzoffset = data['tzoffset']

                # Flatten the JSON and create a dataframe
                df = pd.json_normalize(data, record_path=['days', 'hours'],
                                       meta=[['days', 'datetime'],
                                             ['days', 'datetimeEpoch'],
                                             ['days', 'tempmax'],
                                             ['days', 'tempmin'],
                                             ['days', 'temp'],
                                             ['days', 'feelslikemax'],
                                             ['days', 'feelslikemin'],
                                             ['days', 'humidity'],
                                             ['days', 'dew'],
                                             ['days', 'precip'],
                                             ['days', 'precipprob'],
                                             ['days', 'precipcover'],
                                             ['days', 'snow'],
                                             ['days', 'snowdepth'],
                                             ['days', 'windgust'],
                                             ['days', 'windspeed'],
                                             ['days', 'winddir'],
                                             ['days', 'pressure'],
                                             ['days', 'cloudcover'],
                                             ['days', 'visibility'],
                                             ['days', 'solarradiation'],
                                             ['days', 'solarenergy'],
                                             ['days', 'uvindex'],
                                             ['days', 'severerisk'],
                                             ['days', 'sunrise'],
                                             ['days', 'sunriseEpoch'],
                                             ['days', 'sunset'],
                                             ['days', 'sunsetEpoch'],
                                             ['days', 'moonphase'],
                                             ['days', 'conditions'],
                                             ['days', 'description'],
                                             ['days', 'icon'],
                                             ['days', 'source'],
                                             ['days', 'preciptype'],
                                             ['days', 'stations'],
                                             ])

                # Add the city address and timezone as separate columns
                df['city_address'] = city_address
                df['city_timezone'] = city_timezone
                df['city_latitude'] = city_latitude
                df['city_longitude'] = city_longitude
                df['city_resolvedAddress'] = city_resolvedAddress
                df['city_address'] = city_address
                df['city_timezone'] = city_timezone
                df['city_tzoffset'] = city_tzoffset

                new_names2 = ['day_' + col if 'hours_' not in col and 'city_' not in col else col for col in df.columns]
                df.columns = new_names2

                new_names = {col: col.replace('day_days.', 'hour_') if 'days.' in col else col for col in df.columns}
                df = df.rename(columns=new_names)
                df['day_datetime'] = df['hour_datetime']
                df.drop(['hour_datetime'], axis = 1, inplace = True)
                dfs.append(df)

    df_combined = pd.concat(dfs, axis=0)
    df_combined.to_csv("./scripts/1_data_scrappers/results/history_weather.csv", index=False, encoding='cp1251')

    return df


def check_for_weather_api_token(request_body_json_data):
    if request_body_json_data.get("token") is None:
        raise InvalidUsage("token is required", status_code=400)


def check_for_hours_property(request_body_json_data):
    if request_body_json_data.get("hours") is None:
        request_body_json_data["hours"] = DEFAULT_HOURS


def date_formatter(date, date_format):
    return date.strftime(date_format)


def get_history_from_weather_api(city_name: str, start_date: str, end_date: str):
    url_full = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city_name},Ukraine/{start_date}/{end_date}?unitGroup=metric&aggregateHours=1&key=D27LXDWC3D973WLMLD8RPMZHG&contentType=json&includeAstronomy=true"
    print(url_full)

    response = requests.request("GET", url_full)
    return response.json()


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


def get_city_weather(start_date, end_date, city_name, forecast_hours=DEFAULT_HOURS):
    request_time = dt.datetime.now(dt.timezone.utc)

    forecast_day_start_time_date = date_formatter(request_time, DT_FORMATTING)

    full_weather_details = get_history_from_weather_api(
        city_name, start_date, end_date
    )

    formatted_weather_forecast = format_history_from_weather_api(
        full_weather_details
    )

    return formatted_weather_forecast


def load_regions():
    with open("./external_data/additions/regions.csv", newline="") as csvfile:
        data = csv.DictReader(csvfile)
        center_city_en_array = []
        for row in data:
            center_city_en_array.append(row["center_city_en"])
        return center_city_en_array


def floor_datetime_to_h(time):
    return time.replace(minute=0, second=0, microsecond=0)


if __name__ == "__main__":
    regions = load_regions()
    format_history_from_weather_api()
