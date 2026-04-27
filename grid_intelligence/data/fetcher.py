"""
fetcher.py
==========
Data fetching pipeline for the Grid Intelligence project.

Fetches electricity market data, weather data, and commodity prices
from multiple sources and consolidates them into a single dataset.

Sources:
    - ENTSO-E: Day-ahead prices, generation, load, wind
    - Open-Meteo ERA5: Observed historical weather
    - Open-Meteo Forecast: Weather forecast (+3 days)
    - Yahoo Finance: TTF, WTI, Brent, Henry Hub gas prices

Storage:
    - Development (ENV=development): CSV file at raw_data/consolidated_full.csv
    - Production  (ENV=production):  BigQuery table grid_intelligence.consolidated

Usage:
    # Delta fetch (daily update)
    python fetcher.py --mode delta

    # Full fetch (first time)
    python fetcher.py --mode full --start 2018-10-01
"""

import argparse
import os
import pandas as pd
import yfinance as yf
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timezone
from google.cloud import bigquery
import pandas_gbq

from grid_intelligence.params import (
    ENTSOE_API_KEY, DATA_DIR,
    DELTA_OVERLAP_DAYS, FORECAST_DAYS,
    RENEWABLE, NON_RENEWABLE,
    ENV, GCP_PROJECT, BQ_TABLE_ID
)


class EntsoeSource:
    """
    Fetches electricity market data from the ENTSO-E Transparency Platform.
    Uses the entsoe-py library to query the ENTSO-E REST API.
    All data is resampled to 15-minute resolution.
    """

    def __init__(self, api_key: str):
        from entsoe import EntsoePandasClient
        self.client = EntsoePandasClient(api_key=api_key)

    def fetch_prices(self, start: pd.Timestamp, end: pd.Timestamp, country: str) -> pd.DataFrame:
        """
        Fetch day-ahead electricity prices (Sequence 1, hourly).
        Resampled to 15min using .first() — price is constant within the hour.

        Returns:
            DataFrame with column 'price' in EUR/MWh
        """
        prices = self.client.query_day_ahead_prices(country, start=start, end=end)
        prices = prices.resample('15min').first()
        prices.name = 'price'
        return prices.to_frame()

    def fetch_generation(self, start: pd.Timestamp, end: pd.Timestamp, country: str) -> pd.DataFrame:
        """
        Fetch actual electricity generation by source type.
        Splits into renewable and non-renewable using PSR type classification.
        Resampled to 15min using mean.

        Returns:
            DataFrame with columns:
                - generation: total MW
                - generation_renewable: MW from renewable sources
                - generation_non_renewable: MW from non-renewable sources
        """
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
        """
        Fetch actual total load (consumption) in MW.
        Resampled to 15min using mean.

        Returns:
            DataFrame with column 'consumption' in MW
        """
        load = self.client.query_load(country, start=start, end=end)
        load = load.resample('15min').mean()
        load.columns = ['consumption']
        return load

    def fetch_wind(self, start: pd.Timestamp, end: pd.Timestamp, country: str) -> pd.DataFrame:
        """
        Fetch onshore wind generation (PSR type B19) in MW.
        Resampled to 15min using mean.

        Returns:
            DataFrame with column 'wind_onshore' in MW
        """
        wind = self.client.query_generation(country, start=start, end=end, psr_type='B19')
        wind = wind.resample('15min').mean()
        wind = wind.iloc[:, 0].rename('wind_onshore')
        return wind.to_frame()


class WeatherSource:
    """
    Fetches weather data from Open-Meteo API.

    Two data sources depending on the time range:
        - Past (until today):  ERA5 Reanalysis Archive — real observed data
        - Future (+3 days):    Forecast API — predicted values

    Location: Germany centroid (lat=51.1657, lon=10.4515)
    All data resampled from hourly to 15min via linear interpolation.
    """

    LAT = 51.1657  # Germany centroid latitude
    LON = 10.4515  # Germany centroid longitude
    HOURLY_PARAMS = [
        'temperature_2m',       # Temperature at 2m height
        'relative_humidity_2m', # Relative humidity at 2m
        'cloud_cover',          # Total cloud cover
        'shortwave_radiation',  # Solar radiation W/m²
        'wind_speed_10m'        # Wind speed at 10m height
    ]

    def __init__(self):
        # Cache responses for 1 hour to avoid redundant API calls
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.om = openmeteo_requests.Client(session=retry_session)

    def _parse_response(self, response, suffix: str) -> pd.DataFrame:
        """
        Parse Open-Meteo API response into a DataFrame.

        Args:
            response: Open-Meteo API response object
            suffix: Column name suffix ('_observed' or '_forecast')

        Returns:
            DataFrame with weather columns at 15min resolution
        """
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
        # Upsample from hourly to 15min using linear interpolation
        return df.resample('15min').interpolate(method='linear').ffill()

    def _fetch_archive(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch real observed weather from ERA5 Reanalysis Archive.
        Columns suffixed with '_observed'.
        """
        params = {
            'latitude': self.LAT,
            'longitude': self.LON,
            'hourly': self.HOURLY_PARAMS,
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'timezone': 'UTC'
        }
        response = self.om.weather_api(
            "https://archive-api.open-meteo.com/v1/archive", params=params
        )[0]
        return self._parse_response(response, '_observed')

    def _fetch_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch weather forecast from Open-Meteo Forecast API.
        Available up to +16 days ahead. Columns suffixed with '_forecast'.
        """
        params = {
            'latitude': self.LAT,
            'longitude': self.LON,
            'hourly': self.HOURLY_PARAMS,
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'timezone': 'UTC'
        }
        response = self.om.weather_api(
            "https://api.open-meteo.com/v1/forecast", params=params
        )[0]
        return self._parse_response(response, '_forecast')

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch weather data for the given time range.
        Automatically splits between archive and forecast based on today's date.

        Args:
            start: Start timestamp (UTC)
            end: End timestamp (UTC)

        Returns:
            DataFrame with _observed columns for past and _forecast columns for future
        """
        today = pd.Timestamp.now(tz='UTC').normalize()

        if end <= today:
            # Entirely in the past — use ERA5 archive
            return self._fetch_archive(start, end)
        elif start >= today:
            # Entirely in the future — use forecast
            return self._fetch_forecast(start, end)
        else:
            # Mixed — split at today
            past   = self._fetch_archive(start, today)
            future = self._fetch_forecast(today, end)
            return pd.concat([past, future])


class GasSource:
    """
    Fetches commodity prices from Yahoo Finance.
    All prices are daily and forward-filled to 15min resolution.

    Tickers:
        TTF=F  — TTF Natural Gas (European benchmark, EUR/MWh)
        CL=F   — WTI Crude Oil (US benchmark, USD/barrel)
        BZ=F   — Brent Crude Oil (global benchmark, USD/barrel)
        NG=F   — Henry Hub Natural Gas (US benchmark, USD/MMBtu)
    """

    TICKERS = {
        'TTF=F':  'ttf_gas',
        'CL=F':   'wti_oil',
        'BZ=F':   'brent_oil',
        'NG=F':   'natural_gas'
    }

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch all commodity prices and resample to 15min via forward-fill.

        Returns:
            DataFrame with columns: ttf_gas, wti_oil, brent_oil, natural_gas
        """
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
            # Forward-fill daily prices to 15min — price persists until next trading day
            data = data.resample('15min').ffill()
            data.index.name = 'datetime_utc'
            dfs.append(data)
        return pd.concat(dfs, axis=1)


class DataFetcher:
    """
    Main orchestrator for the data pipeline.

    Coordinates fetching from all sources (ENTSO-E, Weather, Gas),
    merges them into a single DataFrame, and handles storage.

    Storage modes (controlled by ENV variable):
        development: Reads/writes CSV at raw_data/consolidated_full.csv
        production:  Reads/writes BigQuery table grid_intelligence.consolidated

    Delta fetch logic:
        - Reads existing data
        - Fetches last DELTA_OVERLAP_DAYS as overlap buffer
        - Extends FORECAST_DAYS into the future for weather forecast
        - Merges, deduplicates, and saves
    """

    FULL_PATH = 'consolidated_full.csv'

    def __init__(self, entsoe_api_key: str = ENTSOE_API_KEY, output_path: str = DATA_DIR):
        self.entsoe      = EntsoeSource(entsoe_api_key)
        self.weather     = WeatherSource()
        self.gas         = GasSource()
        self.output_path = output_path
        self.full_path   = f'{output_path}/{self.FULL_PATH}'
        self.env         = ENV

        # Initialize BigQuery client only in production
        if self.env == 'production':
            self.bq_client = bigquery.Client(project=GCP_PROJECT)

    def _save(self, df: pd.DataFrame):
        """
        Save DataFrame to storage backend.
        Development: CSV file
        Production: BigQuery table (replace)
        """
        if self.env == 'production':
            pandas_gbq.to_gbq(df, BQ_TABLE_ID, project_id=GCP_PROJECT, if_exists='replace')
            print(f'Saved to BigQuery: {BQ_TABLE_ID}')
        else:
            df.to_csv(self.full_path)
            print(f'Saved to CSV: {self.full_path}')

    def _load(self, tail: int = None) -> pd.DataFrame:
        """
        Load DataFrame from storage backend.

        Args:
            tail: If set, only return the last N rows (faster for large datasets)

        Returns:
            DataFrame with datetime_utc as index (UTC timezone)
        """
        if self.env == 'production':
            if tail:
                # Use LIMIT for efficiency — only fetch what's needed
                query = f'SELECT * FROM `{BQ_TABLE_ID}` ORDER BY datetime_utc DESC LIMIT {tail}'
            else:
                query = f'SELECT * FROM `{BQ_TABLE_ID}` ORDER BY datetime_utc'
            df = self.bq_client.query(query).to_dataframe()
            df = df.set_index('datetime_utc')
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            print(f'Loaded from BigQuery: {df.shape}')
            return df
        else:
            if not os.path.exists(self.full_path):
                raise FileNotFoundError(
                    f'{self.full_path} not found. Run fetch_full first.'
                )
            df = pd.read_csv(self.full_path, index_col=0, parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            if tail:
                return df.tail(tail)
            return df

    @staticmethod
    def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame index to UTC timezone with name 'datetime_utc'.
        Handles MultiIndex by taking the first level.
        """
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.get_level_values(0)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        df.index.name = 'datetime_utc'
        return df

    def _fetch_range(self, start: str, end: str, country: str = 'DE_LU') -> pd.DataFrame:
        """
        Fetch all data sources for a given time range and merge into one DataFrame.

        The merge uses weather as the base (outer join with prices to extend
        into the future), and left joins for all other sources.
        This ensures future rows exist for weather forecast even when
        ENTSO-E data is not yet available.

        Args:
            start: Start date string YYYY-MM-DD
            end: End date string YYYY-MM-DD
            country: ENTSO-E country/zone code

        Returns:
            Merged DataFrame with all 25 feature columns
        """
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

        # Weather is the base — outer join with prices extends into future
        df = weather \
            .join(prices,     how='outer') \
            .join(generation, how='left') \
            .join(load,       how='left') \
            .join(wind,       how='left') \
            .join(gas,        how='left')

        return df

    def fetch_full(self, start: str, end: str, country: str = 'DE_LU') -> pd.DataFrame:
        """
        Full historical fetch from start to end date.
        Use this only once to initialize the dataset.

        Args:
            start: Start date YYYY-MM-DD (e.g. '2018-10-01')
            end: End date YYYY-MM-DD
            country: ENTSO-E country/zone code (default: DE_LU)
        """
        print(f'Full fetch from {start} to {end}...')
        df = self._fetch_range(start, end, country)
        self._save(df)
        print(f'Saved: {df.shape[0]} rows')
        return df

    def fetch_delta(self, country: str = 'DE_LU') -> pd.DataFrame:
        """
        Incremental update — fetches only new data since last update.

        Logic:
            1. Load existing data
            2. Find last date
            3. Fetch from (last_date - DELTA_OVERLAP_DAYS) to (today + FORECAST_DAYS)
            4. Merge with existing, deduplicate (keep latest), save

        The overlap buffer ensures no gaps if the previous fetch had issues.
        The forecast window adds future weather data for prediction use.

        Args:
            country: ENTSO-E country/zone code (default: DE_LU)
        """
        existing = self._load()
        last_date = existing.index.max()
        start = (last_date - pd.Timedelta(days=DELTA_OVERLAP_DAYS)).strftime('%Y-%m-%d')
        end = (datetime.now(timezone.utc) + pd.Timedelta(days=FORECAST_DAYS)).strftime('%Y-%m-%d')

        print(f'Delta fetch from {start} to {end}...')
        delta = self._fetch_range(start, end, country)

        # Log any new columns added since last fetch
        new_cols = [c for c in delta.columns if c not in existing.columns]
        if new_cols:
            print(f'  → New columns detected: {new_cols}')

        # Merge and deduplicate — keep latest values for overlapping rows
        df_all = pd.concat([existing, delta]).sort_index()
        df_all = df_all[~df_all.index.duplicated(keep='last')]
        self._save(df_all)
        print(f'Updated: {df_all.shape[0]} rows')
        return df_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch and consolidate energy data')
    parser.add_argument('--start',          required=False,
                        help='Start date YYYY-MM-DD (required for full fetch)')
    parser.add_argument('--end',            default=datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                        help='End date YYYY-MM-DD (default: today)')
    parser.add_argument('--country',        default='DE_LU',
                        help='ENTSO-E country/zone code (default: DE_LU)')
    parser.add_argument('--output_path',    default=DATA_DIR,
                        help='Output path for CSV (development only)')
    parser.add_argument('--entsoe_api_key', default=ENTSOE_API_KEY,
                        help='ENTSO-E API key (default: from .env)')
    parser.add_argument('--mode',           default='delta', choices=['full', 'delta'],
                        help='Fetch mode: full (initial) or delta (incremental)')
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
