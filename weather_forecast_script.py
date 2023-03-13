from flask import Flask, jsonify, request
import datetime as dt
import requests
import os

app = Flask(__name__)
app.config.from_pyfile(os.path.join(".", "app.conf"), silent=False)

DT_FORMATTING = "%Y-%m-%dT%H:%M:%S"
DEFAULT_HOURS = 12


def formate_forecast_from_weather_api(request_time, location_name, response, forecast_hours):
    location_data = response.get("locations")[location_name]
    location_forecast_values = (location_data['values'])[1:forecast_hours]

    weather_resp = {
        "forecast_request_time": request_time,
        "forecast_location_name": location_name,
        "forecast_hourly_" + str(forecast_hours) + "h_from_now": location_forecast_values,
    }
    return weather_resp


def check_for_weather_api_token(request_body_json_data):
    if request_body_json_data.get("token") is None:
        raise InvalidUsage("token is required", status_code=400)


def check_for_hours_property(request_body_json_data):
    if request_body_json_data.get("hours") is None:
        request_body_json_data["hours"] = DEFAULT_HOURS


def date_formatter(date, date_format):
    return date.strftime(date_format)


def get_forecast_from_weather_api(city_name: str, forecast_start_time: str):
    cross_vis_api_key = app.config.get("CROSS_VIS_WEATHER_API_KEY")

    url_base = "https://weather.visualcrossing.com/VisualCrossingWebServices"
    url_service = "/rest/services/weatherdata/forecast"
    url_params_extras = "?location=" + city_name + \
                        "&unitGroup=metric&aggregateHours=1" \
                        "&key=" + cross_vis_api_key + \
                        "&startDateTime=" + forecast_start_time + \
                        "&contentType=json"

    response = requests.request("GET", url_base + url_service + url_params_extras)
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


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route(
    "/content/api/v1/forecast",
    methods=["POST"],
)
def city_weather_endpoint():
    request_body_json_data = request.get_json()

    check_for_weather_api_token(request_body_json_data)
    check_for_hours_property(request_body_json_data)

    city_name = request_body_json_data.get("location")
    forecast_hours = request_body_json_data.get("hours")

    request_time = dt.datetime.now()

    forecast_day_start_time_date = date_formatter(request_time, DT_FORMATTING)

    full_hourly_weather_details = get_forecast_from_weather_api(city_name,
                                                                forecast_day_start_time_date
                                                                )

    formatted = formate_forecast_from_weather_api(request_time, city_name, full_hourly_weather_details, forecast_hours)

    return formatted
