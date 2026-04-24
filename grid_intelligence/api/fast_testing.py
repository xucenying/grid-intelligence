from fastapi import FastAPI
from datetime import datetime, timezone, timedelta
from grid_intelligence.services.entsoe_client import fetch_day_ahead_prices, parse_prices

app = FastAPI()


@app.get("/entsoe/prices")
def get_prices():

    try:
        # 🕒 Time window (UTC aligned)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=1)

        start_str = start.strftime("%Y%m%d%H%M")
        end_str = end.strftime("%Y%m%d%H%M")

        # 📡 Fetch ENTSO-E data
        xml_data = fetch_day_ahead_prices(start_str, end_str)

        # 🚨 Handle ENTSO-E "no data" responses safely
        if "<Reason>" in xml_data:
            return {
                "status": "no_data",
                "message": "ENTSO-E returned no matching data for this request",
                "raw": xml_data[:1000]
            }

        # 📊 Parse safely
        df = parse_prices(xml_data)

        # 🚨 Extra safety: empty dataframe check
        if df.empty:
            return {
                "status": "no_data",
                "message": "Parsed successfully but no price points found"
            }

        return df.to_dict(orient="records")

    except Exception as e:
        # 🛡️ Never crash API
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/")
def root():
    return {"message": "API is working 🚀"}
