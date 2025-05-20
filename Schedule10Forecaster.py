"""
End-to-end pipeline for Dominion Schedule-10 day-type scraping, temperature fetching, and two-week forecasting
with dynamic hyperparameter tuning for the Random Forest model.

Usage:
  python pipeline.py          # Scrape data, tune model, and forecast
  python pipeline.py --scrape # Only scrape data (day-types & temperatures)
  python pipeline.py --model  # Only tune & forecast (requires existing CSVs)
"""
import argparse
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np

# Scraping dependencies
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC

# Temperature fetching
from meteostat import Point, Daily

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from pandas.tseries.holiday import USFederalHolidayCalendar

# --------------------------------------
# Configuration
# --------------------------------------
START_DATE    = datetime(2020, 1, 1)
END_DATE      = datetime.now()
HORIZON       = 14
N_SPLITS      = 5
RAND_ITERS    = 20
MAPPING       = {'A': 0, 'B': 1, 'C': 2}
DAYS_CSV      = 'schedule_10_day_types.csv'
TEMP_CSV      = 'va_nc_daily_tmax.csv'

# Default hyperparameter search space
DEFAULT_PARAM_DISTS = {
    'n_estimators':    [100, 200, 500],
    'max_depth':       [5, 10, 20, None],
    'min_samples_leaf':[1, 2, 5],
    'class_weight':    ['balanced', 'balanced_subsample'],
    'max_features':    ['sqrt', 'log2', 0.5]
}

# --------------------------------------
# Day-Type Scraper
# --------------------------------------
class DayTypeScraper:
    def __init__(self, start_date, end_date, out_csv=DAYS_CSV):
        self.start = start_date
        self.end = end_date
        self.out_csv = out_csv

    def scrape(self):
        # Build list of year/month to iterate
        months = []
        cur = self.start.replace(day=1)
        while cur <= self.end:
            months.append((cur.year, cur.month))
            cur = (cur + pd.DateOffset(months=1))

        # Launch headless Chrome
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        driver.get("https://www.dominionenergy.com/virginia/rates-and-tariffs/schedule-10-data")

        # Accept cookies
        try:
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            ).click()
        except:
            pass

        # Wait for calendar header
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CLASS_NAME, 'dom-cal-date'))
        )

        def get_header():
            txt = driver.find_element(By.CLASS_NAME, 'dom-cal-date').text
            return datetime.strptime(txt, '%B %Y')

        def go_to_month(year, month):
            hdr = get_header()
            if hdr.year != year:
                sel = Select(driver.find_element(By.ID, 'DateRangeSelection'))
                sel.select_by_value(str(year))
                WebDriverWait(driver, 10).until(lambda d: get_header().year == year)
            start_time = datetime.now().timestamp()
            while True:
                hdr = get_header()
                if hdr.month == month:
                    break
                if datetime.now().timestamp() - start_time > 60:
                    raise RuntimeError(f"Timeout navigating to {month}/{year}")
                btn_id = 'NextMonth' if hdr.month < month else 'PrevMonth'
                driver.find_element(By.ID, btn_id).click()
                pd.sleep(0.3)

        records = []
        for year, month in months:
            go_to_month(year, month)
            cells = driver.find_elements(By.CSS_SELECTOR, '.dom-calendar-col.dom-cal-col-1')
            for cell in cells:
                try:
                    day = int(cell.find_element(By.CLASS_NAME, 'dom-cal-day').text)
                except:
                    continue
                try:
                    cls = cell.find_element(By.CLASS_NAME, 'dom-event').text
                except:
                    cls = ''
                date = datetime(year, month, day)
                if self.start <= date <= self.end:
                    records.append({
                        'Date': date.strftime('%Y-%m-%d'),
                        'Classification': cls
                    })

        driver.quit()
        pd.DataFrame(records).to_csv(self.out_csv, index=False)
        print(f"Saved day-types to {self.out_csv}")

# --------------------------------------
# Temperature Fetcher
# --------------------------------------
class TemperatureFetcher:
    def __init__(self, start_date, end_date, out_csv=TEMP_CSV):
        self.start = start_date
        self.end = end_date
        self.out_csv = out_csv

    def fetch(self):
        va_point = Point(37.5407, -77.4360, 48)  # Richmond, VA
        nc_point = Point(35.7796, -78.6382, 96)  # Raleigh, NC

        df_va = Daily(va_point, self.start, self.end).fetch()[['tmax']]
        df_va.rename(columns={'tmax':'VA_tmax'}, inplace=True)

        df_nc = Daily(nc_point, self.start, self.end).fetch()[['tmax']]
        df_nc.rename(columns={'tmax':'NC_tmax'}, inplace=True)

        df = df_va.join(df_nc, how='outer')
        df.reset_index().rename(columns={'time':'Date'}).to_csv(self.out_csv, index=False)
        print(f"Saved temperatures to {self.out_csv}")

# --------------------------------------
# Day-Type Model with Dynamic Hyperparams
# --------------------------------------
class DayTypeModel:
    def __init__(
        self,
        days_csv=DAYS_CSV,
        temp_csv=TEMP_CSV,
        horizon=HORIZON,
        n_splits=N_SPLITS,
        rand_iters=RAND_ITERS,
        mapping=MAPPING,
        param_distributions=None
    ):
        self.days_csv = days_csv
        self.temp_csv = temp_csv
        self.horizon = horizon
        self.n_splits = n_splits
        self.rand_iters = rand_iters
        self.mapping = mapping
        self.param_distributions = param_distributions or DEFAULT_PARAM_DISTS

    def load_data(self):
        # Load day-types
        df_days = pd.read_csv(self.days_csv)
        df_days['Date'] = pd.to_datetime(df_days['Date'])

        # Load temperatures
        df_temp = pd.read_csv(self.temp_csv)
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])
        df_temp.set_index('Date', inplace=True)
        df_temp['DayOfYear'] = df_temp.index.dayofyear

        # Build climatology
        climatology = df_temp.groupby('DayOfYear')[['VA_tmax','NC_tmax']].mean()

        # Merge
        df = pd.merge(df_days, df_temp.reset_index(), on='Date', how='inner')
        df['ClassEncoded'] = df['Classification'].map(self.mapping)
        df['DayOfWeek'] = df['Date'].dt.weekday
        df['Month'] = df['Date'].dt.month

        # Lags
        for lag in [1,2,3]:
            df[f'Lag_{lag}'] = df['ClassEncoded'].shift(lag)

        # Holiday flag
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=df['Date'].min(), end=df['Date'].max())
        df['IsHoliday'] = df['Date'].isin(holidays).astype(int)

        # Create anomaly and cyclical features
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['VA_anom'] = df['VA_tmax'] - df['DayOfYear'].map(climatology['VA_tmax'])
        df['NC_anom'] = df['NC_tmax'] - df['DayOfYear'].map(climatology['NC_tmax'])
        df['Month_sin'] = np.sin(2*np.pi*df['Month']/12)
        df['Month_cos'] = np.cos(2*np.pi*df['Month']/12)
        df['DoW_sin']   = np.sin(2*np.pi*df['DayOfWeek']/7)
        df['DoW_cos']   = np.cos(2*np.pi*df['DayOfWeek']/7)

        df_model = df.dropna().reset_index(drop=True)

        # Store
        self.df_model = df_model
        self.df_temp = df_temp
        self.climatology = climatology
        self.holidays = holidays

        self.feature_cols = [
            'VA_tmax','VA_tmax_lag1','VA_tmax_lag2','VA_tmax_lag3',
            'NC_tmax','NC_tmax_lag1','NC_tmax_lag2','NC_tmax_lag3',
            'VA_anom','NC_anom',
            'Month_sin','Month_cos','DoW_sin','DoW_cos',
            'Lag_1','Lag_2','Lag_3','IsHoliday'
        ]

    def tune_and_forecast(self):
        # Prepare data
        X = self.df_model[self.feature_cols]
        y = self.df_model['ClassEncoded']

        # Time-series CV
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.horizon)

        # Randomized-search
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions=self.param_distributions,
            n_iter=self.rand_iters,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        search.fit(X, y)
        print("Best RF params:", search.best_params_)
        print(f"Best CV accuracy: {search.best_score_:.2%}")

        # Retrain best estimator on all data
        best_rf = search.best_estimator_
        best_rf.fit(X, y)

        # Forecast helper
        def forecast_horizon(hist_df):
            hist = hist_df.sort_values('Date').reset_index(drop=True)
            last = hist['Date'].iloc[-1]
            lagq = deque(hist['ClassEncoded'].iloc[-3:].tolist(), maxlen=3)
            preds = []
            for i in range(1, self.horizon+1):
                d = last + timedelta(days=i)
                doy = d.timetuple().tm_yday
                if d in self.df_temp.index:
                    va, nc = self.df_temp.loc[d, ['VA_tmax','NC_tmax']]
                else:
                    avg = self.climatology.loc[doy]
                    va, nc = avg['VA_tmax'], avg['NC_tmax']
                feat = {
                    'VA_tmax': va,
                    'VA_tmax_lag1': lagq[-1],
                    'VA_tmax_lag2': lagq[-2],
                    'VA_tmax_lag3': lagq[-3],
                    'NC_tmax': nc,
                    'NC_tmax_lag1': lagq[-1],
                    'NC_tmax_lag2': lagq[-2],
                    'NC_tmax_lag3': lagq[-3],
                    'VA_anom': va - self.climatology.at[doy,'VA_tmax'],
                    'NC_anom': nc - self.climatology.at[doy,'NC_tmax'],
                    'Month_sin': np.sin(2*np.pi*d.month/12),
                    'Month_cos': np.cos(2*np.pi*d.month/12),
                    'DoW_sin': np.sin(2*np.pi*d.weekday()/7),
                    'DoW_cos': np.cos(2*np.pi*d.weekday()/7),
                    'Lag_1': lagq[-1],
                    'Lag_2': lagq[-2],
                    'Lag_3': lagq[-3],
                    'IsHoliday': int(d in self.holidays)
                }
                p = best_rf.predict(pd.DataFrame([feat]))[0]
                lagq.append(p)
                preds.append(p)
            return preds

        # Generate forecast
        past = self.df_model[['Date','ClassEncoded']]
        enc_preds = forecast_horizon(past)
        rev_map = {v: k for k, v in self.mapping.items()}
        future_dates = [self.df_model['Date'].max() + timedelta(days=i)
                        for i in range(1, self.horizon+1)]
        df_out = pd.DataFrame({
            'Date': future_dates,
            'Classification': [rev_map[e] for e in enc_preds]
        })
        df_out.to_csv('two_week_forecast_tuned.csv', index=False)
        print("Saved forecast to two_week_forecast_tuned.csv")

# --------------------------------------
# Pipeline Orchestrator
# --------------------------------------
class Pipeline:
    def __init__(self):
        self.scraper = DayTypeScraper(START_DATE, END_DATE)
        self.fetcher = TemperatureFetcher(START_DATE, END_DATE)
        self.model    = DayTypeModel()

    def run_scrape(self):
        self.scraper.scrape()
        self.fetcher.fetch()

    def run_model(self):
        self.model.load_data()
        self.model.tune_and_forecast()

    def run_all(self):
        self.run_scrape()
        self.run_model()

# --------------------------------------
# Main entrypoint
# --------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scrape', action='store_true', help='Run scraping only')
    parser.add_argument('--model', action='store_true', help='Run modeling only')
    args = parser.parse_args()

    pipeline = Pipeline()
    if args.scrape:
        pipeline.run_scrape()
    elif args.model:
        pipeline.run_model()
    else:
        pipeline.run_all()


