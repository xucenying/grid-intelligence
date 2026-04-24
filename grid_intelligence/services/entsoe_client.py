from urllib import response

import requests
import pandas as pd
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from grid_intelligence.params import ENTSOE_API_KEY

API_KEY = ENTSOE_API_KEY

BASE_URL = "https://web-api.tp.entsoe.eu/api"

def fetch_day_ahead_prices(start, end, country_code="DE_LU"):

    params = {
        "securityToken": API_KEY,
        "documentType": "A44",        # Day-ahead prices
        "in_Domain": country_code,
        "out_Domain": country_code,
        "periodStart": start,
        "periodEnd": end
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"ENTSO-E API error: {response.text}")
    print(response.text)
    return response.text  # XML response



def parse_prices(xml_data):

    if not xml_data or "<" not in xml_data:
        raise ValueError("Invalid or empty ENTSO-E response")

    root = ET.fromstring(xml_data)

    prices = []

    for point in root.findall(".//{*}Point"):
        try:
            position_elem = point.find("{*}position")
            price_elem = point.find("{*}price.amount")

            if position_elem is None or price_elem is None:
                continue

            position = int(position_elem.text)
            price = float(price_elem.text)

            prices.append({
                "position": position,
                "price": price
            })

        except Exception:
            continue  # skip bad rows safely

    if not prices:
        raise ValueError("No price data found (check API key or request)")

    return pd.DataFrame(prices)
