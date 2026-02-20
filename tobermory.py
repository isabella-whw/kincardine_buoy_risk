from datetime import datetime, timedelta, timezone
import requests
import pandas as pd

station_id = "5cebf1e43d0f4a073c4bc39d"

now = datetime.now(timezone.utc)
start = now - timedelta(hours=2)

url = f"https://api-iwls.dfo-mpo.gc.ca/api/v1/stations/{station_id}/data"

params = {
    "time-series-code": "wlo",
    "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "to": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
}

r = requests.get(url, params=params)
r.raise_for_status()

df = pd.DataFrame(r.json())

df["eventDate"] = pd.to_datetime(df["eventDate"], utc=True)
df.rename(columns={"value": "water_level_m"}, inplace=True)
df = df.set_index("eventDate")

hourly = df["water_level_m"].resample("1h").agg(["mean", "min"])

last_completed = hourly.iloc[-2]

print("Last completed hour average:", float(last_completed["mean"]))
print("Last completed hour minimum:", float(last_completed["min"]))