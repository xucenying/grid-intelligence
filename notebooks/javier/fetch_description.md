# Grid Intelligence — Dataset Column Description

## Dataset Overview
- **Source file:** `raw_data/consolidated_full.csv`
- **Granularity:** 15 minutes
- **Range:** 2018-10-01 to present (+3 days forecast)
- **Total columns:** 25
- **Timezone:** UTC (all sources aligned)

---

## Target

| Column | Description | Unit | Source |
|--------|-------------|------|--------|
| `price` | Day-ahead electricity price DE-LU (Sequence 1, hourly, forward-filled) | EUR/MWh | ENTSO-E |

---

## Grid (ENTSO-E)

| Column | Description | Unit | Source |
|--------|-------------|------|--------|
| `generation` | Total electricity generation in Germany | MW | ENTSO-E |
| `generation_renewable` | Renewable generation (Solar, Wind, Hydro, Biomass, etc.) | MW | ENTSO-E |
| `generation_non_renewable` | Non-renewable generation (Gas, Coal, Oil, Nuclear) | MW | ENTSO-E |
| `consumption` | Total electricity consumption in Germany | MW | ENTSO-E |
| `wind_onshore` | Onshore wind power generation | MW | ENTSO-E |

---

## Weather — Legacy / Past Forecast (Open-Meteo Historical Forecast)

> ⚠️ Past forecast data. Replaced by `_observed` columns for model training.

| Column | Description | Unit | Source |
|--------|-------------|------|--------|
| `temperature_c` | Temperature | °C | Open-Meteo Historical Forecast |
| `humidity_percent` | Relative humidity | % | Open-Meteo Historical Forecast |
| `cloud_cover_percent` | Cloud cover | % | Open-Meteo Historical Forecast |
| `shortwave_radiation_wm2` | Solar radiation | W/m² | Open-Meteo Historical Forecast |
| `wind_speed_ms` | Wind speed | m/s | Open-Meteo Historical Forecast |

---

## Weather — Observed (ERA5 Reanalysis)

> ✅ Real observed data. Recommended for model training.

| Column | Description | Unit | Source | Timezone |
|--------|-------------|------|--------|----------|
| `temperature_c_observed` | Temperature (real observed) | °C | Open-Meteo ERA5 Archive | UTC |
| `humidity_percent_observed` | Relative humidity (real observed) | % | Open-Meteo ERA5 Archive | UTC |
| `cloud_cover_percent_observed` | Cloud cover (real observed) | % | Open-Meteo ERA5 Archive | UTC |
| `shortwave_radiation_wm2_observed` | Solar radiation (real observed) | W/m² | Open-Meteo ERA5 Archive | UTC |
| `wind_speed_ms_observed` | Wind speed (real observed) | m/s | Open-Meteo ERA5 Archive | UTC |

---

## Weather — Forecast (Open-Meteo Forecast API)

> 🔮 Available up to +3 days ahead. Used for production predictions.

| Column | Description | Unit | Source | Timezone |
|--------|-------------|------|--------|----------|
| `temperature_c_forecast` | Temperature forecast | °C | Open-Meteo Forecast API | UTC |
| `humidity_percent_forecast` | Relative humidity forecast | % | Open-Meteo Forecast API | UTC |
| `cloud_cover_percent_forecast` | Cloud cover forecast | % | Open-Meteo Forecast API | UTC |
| `shortwave_radiation_wm2_forecast` | Solar radiation forecast | W/m² | Open-Meteo Forecast API | UTC |
| `wind_speed_ms_forecast` | Wind speed forecast | m/s | Open-Meteo Forecast API | UTC |

---

## Commodities (Yahoo Finance, daily, forward-filled)

| Column | Description | Unit | Source | Timezone |
|--------|-------------|------|--------|----------|
| `ttf_gas` | TTF natural gas price (European benchmark) | EUR/MWh | Yahoo Finance (TTF=F) | UTC |
| `wti_oil` | WTI crude oil price (US benchmark) | USD/barrel | Yahoo Finance (CL=F) | UTC |
| `brent_oil` | Brent crude oil price (global benchmark) | USD/barrel | Yahoo Finance (BZ=F) | UTC |
| `natural_gas` | Henry Hub natural gas price (US benchmark) | USD/MMBtu | Yahoo Finance (NG=F) | UTC |

---

## Notes

- All timestamps are in **UTC**
- **15-minute resolution** throughout — hourly sources (prices, commodities) are forward-filled
- **Future rows** (up to +3 days): only `_forecast` weather and commodity columns are populated; grid columns (generation, consumption, price) are `NaN`
- Legacy weather columns (`temperature_c`, etc.) should be **dropped** before model training in favor of `_observed`
- `generation` may lag behind other columns by a few hours due to ENTSO-E publication delay
