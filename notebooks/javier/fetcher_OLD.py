import argparse
import os
import pandas as pd
import yfinance as yf
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timezone

from grid_intelligence.params import (
    ENTSOE_API_KEY, DATA_DIR,
    DELTA_OVERLAP_DAYS, FORECAST_DAYS,
    RENEWABLE, NON_RENEWABLE
)


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
        gen = gen.xs('Actual Aggregated', axis=1, level=1)
        gen = gen.resample('15min').mean()
        renewable_cols = [c for c in gen.columns if c in RENEWABLE]
        non_renewable_cols = [c for c in gen.columns if c in NON_RENEWABLE]
        df = pd.DataFrame(index=gen.index)
        df['generation'] = gen.sum(axis=1)
        df['generation_renewable'] = gen[renewable_cols].sum(axis=1)
        df['generation_non_renewable'] = gen[non_renewable_cols].sum(axis=1)
        return df

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
    HOURLY_PARAMS = ['temperature_2m', 'relative_humidity_2m', 'cloud_cover',
                     'shortwave_radiation', 'wind_speed_10m']

    def __init__(self):
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.om = openmeteo_requests.Client(session=retry_session)

    def _parse_response(self, response, suffix) -> pd.DataFrame:
        hourly = response.Hourly()
        col_names = [
            f'temperature_c{suffix}',
            f'humidity_percent{suffix}',
            f'cloud_cover_percent{suffix}',
            f'shortwave_radiation_wm2{suffix}',
            f'wind_speed_ms{suffix}'
        ]
        df = pd.DataFrame({
            col: hourly.Variables(i).ValuesAsNumpy()
            for i, col in enumerate(col_names)
        }, index=pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit='s', utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit='s', utc=True),
            freq='h',
            inclusive='left'
        ))
        df.index.name = 'datetime_utc'
        return df.resample('15min').interpolate(method='linear').ffill()

    def _fetch_archive(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        params = {
            'latitude': self.LAT,
            'longitude': self.LON,
            'hourly': self.HOURLY_PARAMS,
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'timezone': 'UTC'
        }
        response = self.om.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)[0]
        return self._parse_response(response, '_observed')

    def _fetch_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        params = {
            'latitude': self.LAT,
            'longitude': self.LON,
            'hourly': self.HOURLY_PARAMS,
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'timezone': 'UTC'
        }
        response = self.om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
        return self._parse_response(response, '_forecast')

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        today = pd.Timestamp.now(tz='UTC').normalize()
        if end <= today:
            return self._fetch_archive(start, end)
        elif start >= today:
            return self._fetch_forecast(start, end)
        else:
            past   = self._fetch_archive(start, today)
            future = self._fetch_forecast(today, end)
            return pd.concat([past, future])


class GasSource:
    TICKERS = {
        'TTF=F':  'ttf_gas',
        'CL=F':   'wti_oil',
        'BZ=F':   'brent_oil',
        'NG=F':   'natural_gas'
    }

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        dfs = []
        for ticker, col_name in self.TICKERS.items():
            data = yf.download(
                ticker,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                interval='1d',
                progress=False
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data[['Close']].rename(columns={'Close': col_name})
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            else:
                data.index = data.index.tz_convert('UTC')
            data = data.resample('15min').ffill()
            data.index.name = 'datetime_utc'
            dfs.append(data)
        return pd.concat(dfs, axis=1)


class DataFetcher:
    FULL_PATH = 'consolidated_full.csv'

    def __init__(self, entsoe_api_key: str = ENTSOE_API_KEY, output_path: str = DATA_DIR):
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

        print(f'  → Yahoo Finance gas...')
        gas = self._normalize_index(self.gas.fetch(start_ts, end_ts))

        df = weather \
            .join(prices,     how='outer') \
            .join(generation, how='left') \
            .join(load,       how='left') \
            .join(wind,       how='left') \
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
        start = (last_date - pd.Timedelta(days=DELTA_OVERLAP_DAYS)).strftime('%Y-%m-%d')
        end = (datetime.now(timezone.utc) + pd.Timedelta(days=FORECAST_DAYS)).strftime('%Y-%m-%d')

        print(f'Delta fetch from {start} to {end}...')
        delta = self._fetch_range(start, end, country)

        new_cols = [c for c in delta.columns if c not in existing.columns]
        if new_cols:
            print(f'  → New columns: {new_cols}')

        df_all = pd.concat([existing, delta]).sort_index()
        df_all = df_all[~df_all.index.duplicated(keep='last')]
        df_all.to_csv(self.full_path)
        print(f'Updated: {self.full_path} — {df_all.shape[0]} rows')
        return df_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch and consolidate energy data')
    parser.add_argument('--start',          required=False, help='Start date YYYY-MM-DD (required for full fetch)')
    parser.add_argument('--end',            default=datetime.now(timezone.utc).strftime('%Y-%m-%d'))
    parser.add_argument('--country',        default='DE_LU')
    parser.add_argument('--output_path',    default=DATA_DIR)
    parser.add_argument('--entsoe_api_key', default=ENTSOE_API_KEY)
    parser.add_argument('--mode',           default='delta', choices=['full', 'delta'])
    args = parser.parse_args()

    if not args.entsoe_api_key:
        raise ValueError('ENTSOE_API_KEY missing — set it in .env or pass --entsoe_api_key')

    fetcher = DataFetcher(entsoe_api_key=args.entsoe_api_key, output_path=args.output_path)

    if args.mode == 'delta':
        fetcher.fetch_delta(args.country)
    else:
        if not args.start:
            raise ValueError('--start is required for full fetch')
        fetcher.fetch_full(args.start, args.end, args.country)
