import argparse
import os
import pandas as pd
import yfinance as yf
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timezone
from dotenv import load_dotenv

from pathlib import Path
load_dotenv(Path(__file__).parent.parent.parent / '.env')

from pathlib import Path
env_path = Path(__file__).parent.parent.parent / '.env'
print(f"Suche .env in: {env_path}")
print(f"Existiert: {env_path.exists()}")
load_dotenv(env_path)

class EntsoeSource:
    def __init__(self, api_key: str):
        from entsoe import EntsoePandasClient
        self.client = EntsoePandasClient(api_key=api_key)

    def fetch_prices(self, start: pd.Timestamp, end: pd.Timestamp, country: str) -> pd.DataFrame:
        prices = self.client.query_day_ahead_prices(country, start=start, end=end)
        prices = prices.resample('15min').first()
        prices.name = 'price'
        return prices.to_frame()

    def fetch_generation(self, start: pd.Timestamp, end: pd.Timestamp, country: str) -> pd.DataFrame:
        gen = self.client.query_generation(country, start=start, end=end, psr_type=None)
        gen = gen.resample('15min').mean()
        gen = gen.sum(axis=1).rename('generation')
        return gen.to_frame()

    def fetch_load(self, start: pd.Timestamp, end: pd.Timestamp, country: str) -> pd.DataFrame:
        load = self.client.query_load(country, start=start, end=end)
        load = load.resample('15min').mean()
        load.columns = ['consumption']
        return load

    def fetch_wind(self, start: pd.Timestamp, end: pd.Timestamp, country: str) -> pd.DataFrame:
        wind = self.client.query_generation(country, start=start, end=end, psr_type='B19')
        wind = wind.resample('15min').mean()
        wind = wind.iloc[:, 0].rename('wind_onshore')
        return wind.to_frame()


class WeatherSource:
    LAT = 51.1657
    LON = 10.4515

    def __init__(self):
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.om = openmeteo_requests.Client(session=retry_session)

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        today = pd.Timestamp.now(tz='UTC').normalize()

        if end <= today:
            url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        else:
            url = "https://api.open-meteo.com/v1/forecast"

        params = {
            'latitude': self.LAT,
            'longitude': self.LON,
            'hourly': ['temperature_2m', 'relative_humidity_2m', 'cloud_cover',
                       'shortwave_radiation', 'wind_speed_10m'],
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'timezone': 'UTC'
        }

        response = self.om.weather_api(url, params=params)[0]
        hourly = response.Hourly()

        df = pd.DataFrame({
            'temperature_c': hourly.Variables(0).ValuesAsNumpy(),
            'humidity_percent': hourly.Variables(1).ValuesAsNumpy(),
            'cloud_cover_percent': hourly.Variables(2).ValuesAsNumpy(),
            'shortwave_radiation_wm2': hourly.Variables(3).ValuesAsNumpy(),
            'wind_speed_ms': hourly.Variables(4).ValuesAsNumpy()
        }, index=pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit='s', utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit='s', utc=True),
            freq='h',
            inclusive='left'
        ))
        df.index.name = 'datetime_utc'
        df = df.resample('15min').interpolate(method='linear')
        return df


class GasSource:
    TICKER = 'TTF=F'

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        ttf = yf.download(
            self.TICKER,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            interval='1d',
            progress=False
        )
        if isinstance(ttf.columns, pd.MultiIndex):
            ttf.columns = ttf.columns.get_level_values(0)
        ttf = ttf[['Close']].copy()
        ttf.columns = ['ttf_gas']
        if isinstance(ttf.index, pd.MultiIndex):
            ttf.index = ttf.index.get_level_values(0)
        if ttf.index.tz is None:
            ttf.index = ttf.index.tz_localize('UTC')
        else:
            ttf.index = ttf.index.tz_convert('UTC')
        ttf = ttf.resample('15min').ffill()
        ttf.index.name = 'datetime_utc'
        return ttf


class DataFetcher:
    FULL_PATH = 'consolidated_full.csv'

    def __init__(self, entsoe_api_key: str, output_path: str = 'raw_data'):
        self.entsoe      = EntsoeSource(entsoe_api_key)
        self.weather     = WeatherSource()
        self.gas         = GasSource()
        self.output_path = output_path
        self.full_path   = f'{output_path}/{self.FULL_PATH}'

    @staticmethod
    def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.get_level_values(0)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        df.index.name = 'datetime_utc'
        return df

    def _fetch_range(self, start: str, end: str, country: str = 'DE_LU') -> pd.DataFrame:
        start_ts = pd.Timestamp(start, tz='UTC')
        end_ts   = pd.Timestamp(end, tz='UTC')

        print(f'  → ENTSO-E prices...')
        prices = self._normalize_index(self.entsoe.fetch_prices(start_ts, end_ts, country))

        print(f'  → ENTSO-E generation...')
        generation = self._normalize_index(self.entsoe.fetch_generation(start_ts, end_ts, country))

        print(f'  → ENTSO-E load...')
        load = self._normalize_index(self.entsoe.fetch_load(start_ts, end_ts, country))

        print(f'  → ENTSO-E wind...')
        wind = self._normalize_index(self.entsoe.fetch_wind(start_ts, end_ts, country))

        print(f'  → Open-Meteo weather...')
        weather = self._normalize_index(self.weather.fetch(start_ts, end_ts))

        print(f'  → Yahoo Finance TTF gas...')
        gas = self._normalize_index(self.gas.fetch(start_ts, end_ts))

        df = prices \
            .join(generation, how='left') \
            .join(load,       how='left') \
            .join(wind,       how='left') \
            .join(weather,    how='left') \
            .join(gas,        how='left')

        return df

    def fetch_full(self, start: str, end: str, country: str = 'DE_LU') -> pd.DataFrame:
        print(f'Full fetch from {start} to {end}...')
        df = self._fetch_range(start, end, country)
        df.to_csv(self.full_path)
        print(f'Saved: {self.full_path} — {df.shape[0]} rows')
        return df

    def fetch_delta(self, country: str = 'DE_LU') -> pd.DataFrame:
        if not os.path.exists(self.full_path):
            raise FileNotFoundError(f'{self.full_path} not found. Run fetch_full first.')

        existing = pd.read_csv(self.full_path, index_col=0, parse_dates=True)
        if existing.index.tz is None:
            existing.index = existing.index.tz_localize('UTC')

        last_date = existing.index.max()
        start = (last_date - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
        end   = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        print(f'Delta fetch from {start} to {end}...')
        delta = self._fetch_range(start, end, country)

        df_all = pd.concat([existing, delta]).sort_index()
        df_all = df_all[~df_all.index.duplicated(keep='last')]
        df_all.to_csv(self.full_path)
        print(f'Updated: {self.full_path} — {df_all.shape[0]} rows')
        return df_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch and consolidate energy data')
    parser.add_argument('--start',       required=False, help='Start date YYYY-MM-DD (required for full fetch)')
    parser.add_argument('--end',         default=datetime.now(timezone.utc).strftime('%Y-%m-%d'))
    parser.add_argument('--country',     default='DE_LU')
    parser.add_argument('--output_path', default='raw_data')
    parser.add_argument('--entsoe_api_key',  default=os.getenv('ENTSOE_API_KEY'))  # ← geändert
    parser.add_argument('--mode',        default='full', choices=['full', 'delta'])
    args = parser.parse_args()

    if not args.entsoe_api_key:
        raise ValueError('ENTSOE_API_KEY fehlt — setze es in .env oder übergib --entsoe_api_key')

    fetcher = DataFetcher(entsoe_api_key=args.entsoe_api_key, output_path=args.output_path)

    if args.mode == 'delta':
        fetcher.fetch_delta(args.country)
    else:
        if not args.start:
            raise ValueError('--start is required for full fetch')
        fetcher.fetch_full(args.start, args.end, args.country)
