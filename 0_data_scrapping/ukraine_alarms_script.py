import csv
import datetime
import json
import os

import requests

DT_FORMATTING = "%Y-%m-%dT%H:%M:%S"

API_URL = "https://api.ukrainealarm.com/api/v3/"
REGIONS_ENDPOINT = "regions"
ALERTS_REGION_HISTORY_ENDPOINT = "alerts/regionHistory"
ALERTS_ENDPOINT = "alerts"

LUHANSK_ID = 16
CRIMEA_ID = 9999
EXCEPT_REGIONS = [LUHANSK_ID, CRIMEA_ID]

UKRAINE_ALARMS_TOKEN = os.environ.get('UKRAINE_ALARMS_TOKEN')

if UKRAINE_ALARMS_TOKEN is None:
    print("UKRAINE_ALARMS_TOKEN is not set in the environment variables.")


# Utils
def convert_region_name(name):
    return name.replace("область", "").replace("м.", "").strip()


def floor_datetime_to_h(time):
    return time.replace(minute=0, second=0, microsecond=0)


def build_headers(token):
    return {"Authorization": f"{token}"}


def datetime_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()


def pretty_print_json(list):
    for data in list:
        print(json.dumps(data, indent=4, default=datetime_converter))
        print("--------------------")


# API integration

def get_current_alerts(token):
    headers = build_headers(token)
    response = requests.get(API_URL + ALERTS_ENDPOINT, headers=headers)
    response_data = response.json()
    return response_data


def get_last_25_region_alerts(token, region_id):
    headers = build_headers(token)
    response = requests.get(API_URL + ALERTS_REGION_HISTORY_ENDPOINT + f"?regionId={region_id}", headers=headers)
    response_data = response.json()
    return response_data


def get_regions(token):
    headers = build_headers(token)
    response = requests.get(API_URL + REGIONS_ENDPOINT, headers=headers)
    response_data = response.json()
    return response_data


# API response parsing

def parse_states(regions_json):
    parsed_states = []
    for region in regions_json["states"]:
        if region["regionType"] == "State":
            parsed_state = {
                "regionId": int(region["regionId"]),
                "regionName": region["regionName"],
                "regionType": region["regionType"]
            }
            parsed_states.append(parsed_state)
    sorted_states = sorted(parsed_states, key=lambda k: k['regionId'])
    filtered_states = filter(lambda s: s['regionId'] != 0, sorted_states)
    formatted_states = map(lambda s: {**s, "regionName": convert_region_name(s["regionName"])}, filtered_states)
    formatted_states = [{**s, "regionName": "АР Крим" if s["regionId"] == 9999 else s["regionName"]} for s in
                        formatted_states]
    return formatted_states


def parse_current_alerts(alerts_json):
    parsed_alerts = []
    for region_info in alerts_json:
        if region_info["regionType"] == 'State':
            state_alerts = map(lambda s: {**s, "regionId": int(s["regionId"])}, region_info["activeAlerts"])
            parsed_alerts.extend(state_alerts)
    filtered_alerts = filter(lambda s: s['type'] == 'AIR' and s['regionType'] == 'State', parsed_alerts)
    sorted_alerts = sorted(filtered_alerts, key=lambda k: k['regionId'])
    return sorted_alerts


def parse_last_24_hours_alerts(last_25_region_alerts_json, from_date, to_date):
    last_25_alarms = last_25_region_alerts_json[0]['alarms']
    last_25_alarms = map(lambda s: {**s,
                                    "regionId": int(s["regionId"]),
                                    "startDate": datetime.datetime.strptime(s["startDate"], DT_FORMATTING),
                                    "endDate": datetime.datetime.strptime(s["endDate"], DT_FORMATTING),
                                    "regionName": convert_region_name(s["regionName"])
                                    },
                         last_25_alarms)
    last_24_hours_alarms = [alarm for alarm in last_25_alarms if from_date <= alarm['startDate'] <= to_date]
    return last_24_hours_alarms


def calc_current_alarms_count():
    curr_alerts = get_current_alerts(UKRAINE_ALARMS_TOKEN)
    parsed_curr_alerts = parse_current_alerts(curr_alerts)
    filtered_curr_alerts = list(filter(lambda alert: alert['regionId'] not in EXCEPT_REGIONS, parsed_curr_alerts))
    return len(filtered_curr_alerts)


def calculate_alarms(script_start_time):
    # Floor the datetime object to the nearest hour
    script_start_hour = floor_datetime_to_h(script_start_time)

    # Get regions info
    regions = get_regions(UKRAINE_ALARMS_TOKEN)
    states = parse_states(regions)

    # Get Current alarms
    curr_alarms_count = calc_current_alarms_count()

    # Get alarms for last 24 hours for the region
    one_day = datetime.timedelta(days=1)
    previous_day = script_start_time - one_day

    result = []

    for state in list(filter(lambda state: state['regionId'] not in EXCEPT_REGIONS, states)):
        last_25_region_alerts = get_last_25_region_alerts(UKRAINE_ALARMS_TOKEN, state['regionId'])
        last_24_hours_alerts = parse_last_24_hours_alerts(last_25_region_alerts, previous_day, now)
        result.append({
            "ua_alarms_region_id": state['regionId'],
            "region": state['regionName'],
            "events_last_24_hrs": len(last_24_hours_alerts)
        })

    script_start_hour_str = script_start_hour.strftime(DT_FORMATTING)
    script_start_datetime_str = script_start_time.strftime(DT_FORMATTING)

    for dict in result:
        dict.update({
            'alarmed_regions_count': curr_alarms_count,
            'start_hour': script_start_hour_str,
            'day_datetime': script_start_datetime_str
        })

    return result


def save_alarms_data_to_csv(alarms_data, script_start_hour):
    date_str = script_start_hour.strftime("%Y-%m-%d")
    hour_str = script_start_hour.strftime("%H_%M")
    dir_path = f'results/alarms/{date_str}'
    file_path = f'{dir_path}/alarms_{hour_str}.csv'
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = alarms_data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in alarms_data:
            writer.writerow(row)


if __name__ == "__main__":
    now = datetime.datetime.now()
    script_start_time = now

    alarms_data = calculate_alarms(script_start_time)

    script_start_hour = floor_datetime_to_h(script_start_time)
    save_alarms_data_to_csv(alarms_data, script_start_hour)
