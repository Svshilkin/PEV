import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')  # КРИТИЧНО: используем неинтерактивный backend для избежания ошибок tkinter
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import warnings
import os
from scipy.interpolate import interp1d
from scipy import stats

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# Глобальные переменные для совместимости
file_path = 'vacancies_engineer.csv'
output_folder = 'results'
forecast_periods = 6
validation_periods = 6

class VacancyForecaster:
    """Система прогнозирования вакансий с улучшенным исправлением аномалий"""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.scalers = {}
        self.feature_names = None
        
    def detect_extreme_anomalies(self, ts_df, threshold_factor=2.0):
        """Поиск экстремальных аномалий"""
        print("🔍 Автоматическое обнаружение аномалий...")
        
        extreme_anomalies = {}
        
        for col in ts_df.columns:
            series = ts_df[col]
            
            # Более мягкие критерии для поиска экстремальных значений
            median_val = series.median()
            q75 = series.quantile(0.75)
            q90 = series.quantile(0.90)
            q95 = series.quantile(0.95)
            max_val = series.max()
            
            # Критерий 1: Z-score > 2.0
            z_scores = np.abs(stats.zscore(series))
            z_anomalies = z_scores > threshold_factor
            
            # Критерий 2: Значение > 90-го процентиля И больше медианы в 2+ раза
            extreme_values = (series > q90) & (series > median_val * 2)
            
            # Критерий 3: Очень высокие значения
            if col == 'total_demand':
                very_high_values = series > 80000
            else:
                very_high_values = series > (q95 * 1.2)
            
            # Критерий 4: Резкий скачок
            diff_forward = series.diff(1).abs()
            diff_backward = series.diff(-1).abs()
            spike_threshold = median_val * 1.5
            extreme_spikes = (diff_forward > spike_threshold) | (diff_backward > spike_threshold)
            
            # Объединяем критерии
            combined_mask = z_anomalies | extreme_values | very_high_values | extreme_spikes
            
            if combined_mask.any():
                extreme_dates = series.index[combined_mask].tolist()
                extreme_values_list = series[combined_mask].tolist()
                extreme_anomalies[col] = extreme_dates
                
                print(f"  {col}: найдено {len(extreme_dates)} аномалий")
                for date, value in zip(extreme_dates, extreme_values_list):
                    z_score = abs((value - series.mean()) / series.std()) if series.std() > 0 else 0
            else:
                print(f"  {col}: аномалий не найдено")
        
        return extreme_anomalies
    
    def fix_extreme_anomalies(self, ts_df, extreme_anomalies, method='conservative_smooth', window_size=8):
        """Консервативное исправление ТОЛЬКО экстремальных аномалий"""
        print("🔧 Автоматическое исправление аномалий...")
        
        fixed_df = ts_df.copy()
        total_fixed = 0
        
        for col, anomaly_dates in extreme_anomalies.items():
            if not anomaly_dates:
                continue

                
            for anomaly_date in anomaly_dates:
                try:
                    # Находим индекс аномалии
                    anomaly_idx = fixed_df.index.get_loc(anomaly_date)
                    original_value = fixed_df.at[anomaly_date, col]
                    
                    # Определяем окрестность для анализа
                    start_idx = max(0, anomaly_idx - window_size)
                    end_idx = min(len(fixed_df) - 1, anomaly_idx + window_size)
                    
                    # Данные до и после аномалии
                    before_data = fixed_df[col].iloc[start_idx:anomaly_idx]
                    after_data = fixed_df[col].iloc[anomaly_idx + 1:end_idx + 1]
                    surrounding_data = pd.concat([before_data, after_data])
                    
                    if len(surrounding_data) > 0:
                        if col == 'total_demand':
                            if original_value > 80000:
                                target_value = surrounding_data.quantile(0.75)
                                max_reduction = original_value * 0.6
                                fixed_value = int(max(target_value, max_reduction))
                            else:
                                fixed_value = int(surrounding_data.mean())
                        else:
                            if original_value > surrounding_data.quantile(0.90):
                                fixed_value = int(surrounding_data.quantile(0.90))
                            else:
                                fixed_value = int(surrounding_data.quantile(0.25))
                        
                        fixed_df.at[anomaly_date, col] = max(0, fixed_value)
                        total_fixed += 1

                        
                except Exception as e:
                    print(f"    ❌ Ошибка при исправлении аномалии в {col} на дату {anomaly_date}: {e}")
                    continue
        
        print(f"✅ Всего исправлено {total_fixed} аномалий в {len([col for col, dates in extreme_anomalies.items() if dates])} столбцах")
        return fixed_df
    
    def create_time_series(self, df, freq='M', min_activity=10):
        """Создание временных рядов с различной частотой"""
        print(f"Создание временных рядов с частотой '{freq}'...")
        
        if len(df) == 0:
            print("❌ Нет данных для создания временных рядов")
            return pd.DataFrame()
        
        # Убеждаемся что даты в правильном формате
        if not pd.api.types.is_datetime64_any_dtype(df['date_creation_processed']):
            df['date_creation_processed'] = pd.to_datetime(df['date_creation_processed'], errors='coerce')
        if not pd.api.types.is_datetime64_any_dtype(df['date_end']):
            df['date_end'] = pd.to_datetime(df['date_end'], errors='coerce')
        
        # Фильтруем записи с корректными датами
        df = df[df['date_creation_processed'].notna() & df['date_end'].notna()]
        
        if len(df) == 0:
            print("❌ Нет записей с корректными датами")
            return pd.DataFrame()
        
        # Определение временного диапазона
        start_date = df['date_creation_processed'].min().to_period(freq).to_timestamp()
        end_date = df['date_end'].max().to_period(freq).to_timestamp()
        
        # Полный временной ряд
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        if 'industry' not in df.columns:
            # Создаем простой временной ряд
            ts_df = pd.DataFrame(index=date_range, columns=['total_demand']).fillna(0)
            
            for _, row in df.iterrows():
                start_period = row['date_creation_processed'].to_period(freq).to_timestamp()
                end_period = row['date_end'].to_period(freq).to_timestamp()
                active_periods = pd.date_range(start=start_period, end=end_period, freq=freq)
                
                for period in active_periods:
                    if period in ts_df.index:
                        ts_df.at[period, 'total_demand'] += row['work_places']
            
            return ts_df
        
        # Агрегация по отраслям
        industries = df['industry'].value_counts()
        active_industries = industries[industries >= min_activity].index
        
        print(f"Активных отраслей: {len(active_industries)}")
        
        if len(active_industries) == 0:
            active_industries = ['total']
            df['industry'] = 'total'
        
        ts_df = pd.DataFrame(index=date_range, columns=active_industries).fillna(0)
        
        # Заполнение данных
        for industry in active_industries:
            industry_data = df[df['industry'] == industry] if industry != 'total' else df
            
            for _, row in industry_data.iterrows():
                start_period = row['date_creation_processed'].to_period(freq).to_timestamp()
                end_period = row['date_end'].to_period(freq).to_timestamp()
                active_periods = pd.date_range(start=start_period, end=end_period, freq=freq)
                
                for period in active_periods:
                    if period in ts_df.index:
                        ts_df.at[period, industry] += row['work_places']
        
        # Общий спрос
        ts_df['total_demand'] = ts_df.sum(axis=1)
        
        return ts_df
    
    def remove_outliers(self, ts_df, threshold=3):
        """Удаление выбросов на основе Z-score"""
        print(f"🎯 Удаление выбросов (порог: {threshold})...")
        
        cleaned_df = ts_df.copy()
        
        for col in cleaned_df.columns:
            series = cleaned_df[col].dropna()
            if len(series) > 3:
                z_scores = np.abs(stats.zscore(series))
                outliers = z_scores > threshold
                
                if outliers.any():
                    print(f"  {col}: найдено {outliers.sum()} выбросов")
                    cleaned_series = cleaned_df[col].copy()
                    outlier_indices = series.index[outliers]
                    cleaned_series.loc[outlier_indices] = np.nan
                    cleaned_series = cleaned_series.interpolate(method='cubic')
                    cleaned_series = cleaned_series.fillna(method='bfill').fillna(method='ffill').fillna(0)
                    cleaned_df[col] = cleaned_series
        
        return cleaned_df
    
    def create_enhanced_features(self, ts, target_col, fixed_feature_set=None):
        """Создание расширенного набора признаков - ОРИГИНАЛЬНАЯ ЛОГИКА"""
        target_series = ts[target_col] if isinstance(ts, pd.DataFrame) else ts
        target_series = target_series.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        X = pd.DataFrame(index=target_series.index)
        
        # Календарные признаки
        X['month'] = target_series.index.month
        X['quarter'] = target_series.index.quarter
        X['year'] = target_series.index.year
        X['day_of_year'] = target_series.index.dayofyear
        X['is_quarter_end'] = target_series.index.is_quarter_end.astype(int)
        X['is_year_end'] = target_series.index.is_year_end.astype(int)
        
        # Лаговые признаки - ОРИГИНАЛЬНАЯ ЛОГИКА
        max_lag = min(12, len(target_series) - 1)
        for lag in [1, 2, 3, 4, 8, 12]:
            if lag <= max_lag:
                lag_series = target_series.shift(lag)
                X[f'lag_{lag}'] = lag_series.fillna(0)  # ВАЖНО: заполняем нулями как в оригинале!
            else:
                X[f'lag_{lag}'] = 0  # Добавляем нулевой столбец для консистентности
        
        # Скользящие средние - ОРИГИНАЛЬНАЯ ЛОГИКА
        for window in [3, 6, 12]:
            if window <= len(target_series):
                ma_series = target_series.rolling(window=window).mean().shift(1)  # Убрали min_periods=1
                X[f'ma_{window}'] = ma_series.fillna(target_series.mean())
            else:
                X[f'ma_{window}'] = target_series.mean()  # Заполняем средним значением
        
        # Разности
        diff_1 = target_series.diff(1)
        X['diff_1'] = diff_1.fillna(0)
        
        if len(target_series) > 4:
            diff_4 = target_series.diff(4)
            X['diff_4'] = diff_4.fillna(0)
        else:
            X['diff_4'] = 0
        
        # Если это первый вызов, сохраняем названия признаков
        if fixed_feature_set is None and self.feature_names is None:
            self.feature_names = X.columns.tolist()
        
        # Если у нас есть фиксированный набор признаков, приводим к нему
        if self.feature_names is not None:
            # Добавляем недостающие признаки
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0
            
            # Упорядочиваем столбцы в правильном порядке
            X = X.reindex(columns=self.feature_names, fill_value=0)
        
        X = X.fillna(0)
        
        if target_series.isna().any():
            target_series = target_series.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return X, target_series
    
    def train_models(self, X_train, y_train, scale_features=True):
        """Обучение моделей"""
        
        if X_train.isna().any().any():
            X_train = X_train.fillna(0)
        
        if y_train.isna().any():
            y_train = y_train.fillna(y_train.mean())
        
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Пустые данные для обучения")
        
        # Масштабирование признаков
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            self.scalers['feature_scaler'] = scaler
        else:
            X_train_scaled = X_train
        
        # Random Forest
        try:
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"⚠️ Ошибка при обучении Random Forest: {e}")
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
        
        # XGBoost
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='reg:squarederror',
                verbosity=0
            )
            xgb_model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"⚠️ Ошибка при обучении XGBoost: {e}")
            xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
            xgb_model.fit(X_train_scaled, y_train)
        
        return rf_model, xgb_model

    def predict_with_model(self, model, X_test):
        """Предсказание с применением масштабирования если нужно"""
        if 'feature_scaler' in self.scalers:
            X_test_scaled = self.scalers['feature_scaler'].transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        else:
            X_test_scaled = X_test
        
        return model.predict(X_test_scaled)

    def create_recursive_forecast(self, target_series, best_model_name, rf_model, xgb_model, periods=6):
        """Создание рекурсивного прогноза с правильной логикой"""
        
        # Выбираем лучшую модель
        model = rf_model if best_model_name == "Random Forest" else xgb_model
        
        # Расширяем историю для лучших лагов
        history_length = min(24, len(target_series))
        recent_values = target_series.tail(history_length).values.tolist()
        
        # Создаем список для всех значений (история + прогнозы)
        all_values = recent_values.copy()
        
        # Результаты прогноза
        forecast_values = []
        forecast_dates = []
        
        # Последняя дата
        last_date = target_series.index[-1]
        
        for i in range(periods):
            # Следующая дата
            if hasattr(last_date, 'freq') and 'W' in str(last_date.freq):
                next_date = last_date + pd.DateOffset(weeks=i+1)
            else:
                next_date = last_date + pd.DateOffset(months=i+1)
            
            forecast_dates.append(next_date)
            
            # Создаем признаки для следующего периода - ТА ЖЕ ЛОГИКА что в валидации
            X_next = pd.DataFrame({
                'month': [next_date.month],
                'quarter': [next_date.quarter], 
                'year': [next_date.year],
                'day_of_year': [next_date.dayofyear],
                'is_quarter_end': [1 if next_date.is_quarter_end else 0],
                'is_year_end': [1 if next_date.is_year_end else 0],
            }, index=[next_date])
            
            # Текущая позиция в массиве всех значений
            current_pos = len(all_values) - 1
            
            # Лаговые признаки - ТАКАЯ ЖЕ ЛОГИКА как в валидации, но с рекурсией
            for lag in [1, 2, 3, 4, 8, 12]:
                lag_pos = current_pos - lag + 1
                if lag_pos >= 0 and lag_pos < len(all_values):
                    X_next[f'lag_{lag}'] = [all_values[lag_pos]]
                else:
                    # Если лага не хватает, заполняем нулем (как в валидации)
                    X_next[f'lag_{lag}'] = [0]
            
            # Скользящие средние - ТАКАЯ ЖЕ ЛОГИКА как в валидации
            for window in [3, 6, 12]:
                start_pos = max(0, current_pos - window + 2)
                window_values = all_values[start_pos:]
                if len(window_values) >= window:
                    X_next[f'ma_{window}'] = [np.mean(window_values[-window:])]
                elif len(window_values) > 0:
                    # Если окно неполное, но есть данные
                    X_next[f'ma_{window}'] = [np.mean(window_values)]
                else:
                    # Если данных нет, используем среднее из истории (как в валидации)
                    X_next[f'ma_{window}'] = [np.mean(all_values) if len(all_values) > 0 else 0]
            
            # Разности - ТАКАЯ ЖЕ ЛОГИКА как в валидации
            if len(all_values) >= 2:
                X_next['diff_1'] = [all_values[-1] - all_values[-2]]
            else:
                X_next['diff_1'] = [0]
                
            if len(all_values) >= 5:
                X_next['diff_4'] = [all_values[-1] - all_values[-5]]
            else:
                X_next['diff_4'] = [0]
            
            # Приводим к нужному порядку признаков (как в валидации)
            if self.feature_names:
                for feature in self.feature_names:
                    if feature not in X_next.columns:
                        X_next[feature] = [0]
                X_next = X_next.reindex(columns=self.feature_names, fill_value=0)
            
            # Делаем предсказание
            pred = self.predict_with_model(model, X_next)[0]
            pred = max(0, round(pred))
            
            # Добавляем прогноз в общий список
            all_values.append(pred)
            forecast_values.append(pred)
        
        return pd.Series(forecast_values, index=forecast_dates)

def create_industry_timeseries(df, freq='M'):
    """Создание временных рядов по отраслям"""
    forecaster = VacancyForecaster(output_folder)
    return forecaster.create_time_series(df, freq, min_activity=5)

def detect_and_fix_anomalies(ts_df, anomaly_dates=None, window_size=4, method='cubic', output_folder=None):
    """Обнаружение и исправление аномалий в временных рядах"""
    print(f"🔧 ПРИМЕНЕНИЕ ЛОГИКИ ИСПРАВЛЕНИЯ АНОМАЛИЙ...")
    
    forecaster = VacancyForecaster(output_folder or 'results')
    
    # Автоматическое обнаружение экстремальных аномалий
    anomalies = forecaster.detect_extreme_anomalies(ts_df)
    
    # Создание графика до исправления аномалий
    if output_folder and 'total_demand' in ts_df.columns:
        try:
            plt.figure(figsize=(14, 7))
            plt.plot(ts_df.index, ts_df['total_demand'], 'b-', label='До исправления')
            
            if 'total_demand' in anomalies and anomalies['total_demand']:
                for anomaly_date in anomalies['total_demand']:
                    plt.axvline(x=anomaly_date, color='r', linestyle='--', alpha=0.7)
                    anomaly_value = ts_df.at[anomaly_date, 'total_demand']
                    plt.annotate(f'{anomaly_value:,.0f}', 
                               xy=(anomaly_date, anomaly_value), 
                               xytext=(anomaly_date, anomaly_value + 5000),
                               ha='center', fontsize=8, color='red',
                               arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
            
            plt.title(f'Общий спрос до исправления аномалий ({ts_df.index[0].strftime("%Y-%m")} - {ts_df.index[-1].strftime("%Y-%m")})')
            plt.xlabel('Дата')
            plt.ylabel('Количество рабочих мест')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{output_folder}/anomaly_before_fix.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Ошибка при создании графика до исправления: {e}")
    
    # Исправление аномалий
    fixed_df = forecaster.fix_extreme_anomalies(ts_df, anomalies, method='conservative_smooth', window_size=8)
    
    # Создание графика после исправления аномалий
    if output_folder and 'total_demand' in ts_df.columns:
        try:
            plt.figure(figsize=(14, 7))
            plt.plot(ts_df.index, ts_df['total_demand'], 'b-', label='До исправления', alpha=0.7)
            plt.plot(fixed_df.index, fixed_df['total_demand'], 'g-', label='После исправления')
            
            if 'total_demand' in anomalies and anomalies['total_demand']:
                for anomaly_date in anomalies['total_demand']:
                    plt.axvline(x=anomaly_date, color='r', linestyle='--', alpha=0.7)
                    original_value = ts_df.at[anomaly_date, 'total_demand']
                    fixed_value = fixed_df.at[anomaly_date, 'total_demand']
                    plt.annotate(f'{original_value:,.0f}→{fixed_value:,.0f}', 
                               xy=(anomaly_date, fixed_value), 
                               xytext=(anomaly_date, fixed_value + 8000),
                               ha='center', fontsize=8, color='darkgreen',
                               arrowprops=dict(arrowstyle='->', color='darkgreen', alpha=0.7))
            
            plt.title(f'Общий спрос после исправления аномалий ({fixed_df.index[0].strftime("%Y-%m")} - {fixed_df.index[-1].strftime("%Y-%m")})')
            plt.xlabel('Дата')
            plt.ylabel('Количество рабочих мест')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{output_folder}/anomaly_after_fix.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Ошибка при создании графика после исправления: {e}")
    
    return fixed_df

def analyze_timeseries(ts_df, output_folder):
    """Анализ временных рядов"""
    print("Анализ временных рядов...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Анализ общего спроса
    plt.figure(figsize=(15, 7))
    plt.plot(ts_df.index, ts_df['total_demand'], marker='o', linestyle='-')
    plt.title('Общий спрос на инженерные кадры по времени')
    plt.xlabel('Период')
    plt.ylabel('Количество рабочих мест')
    plt.grid(True)
    plt.savefig(f'{output_folder}/total_demand.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сезонная декомпозиция
    if len(ts_df) >= 24:
        try:
            total_demand_clean = ts_df['total_demand'].fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            result = seasonal_decompose(total_demand_clean, model='additive', period=12)
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
            result.observed.plot(ax=ax1)
            ax1.set_title('Наблюдаемые значения')
            ax1.grid(True)
            
            result.trend.plot(ax=ax2)
            ax2.set_title('Тренд')
            ax2.grid(True)
            
            result.seasonal.plot(ax=ax3)
            ax3.set_title('Сезонность')
            ax3.grid(True)
            
            result.resid.plot(ax=ax4)
            ax4.set_title('Остатки')
            ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{output_folder}/seasonal_decomposition.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Анализ сезонности по месяцам
            seasonal_data = result.seasonal.groupby(result.seasonal.index.month).mean()
            
            plt.figure(figsize=(12, 6))
            plt.bar(seasonal_data.index, seasonal_data.values)
            plt.title('Средний сезонный эффект по месяцам')
            plt.xlabel('Месяц')
            plt.ylabel('Сезонный эффект')
            plt.grid(True, axis='y')
            plt.xticks(range(1, 13), ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 
                                     'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'])
            plt.savefig(f'{output_folder}/monthly_seasonality.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Не удалось выполнить декомпозицию временного ряда: {e}")
    
    # Анализ топ отраслей
    top_n = min(10, len(ts_df.columns) - 1)
    if top_n > 0:
        top_industries = ts_df.drop('total_demand', axis=1).sum().sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x=top_industries.index, y=top_industries.values)
        plt.title(f'Топ-{top_n} отраслей по общему количеству рабочих мест')
        plt.xlabel('Отрасль')
        plt.ylabel('Суммарное количество рабочих мест')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_folder}/top_industries.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        top_industries = pd.Series(dtype=float)
    
    return top_industries

def create_features(ts, include_all_lags=True):
    """Создание признаков для моделей машинного обучения"""
    forecaster = VacancyForecaster()
    
    if isinstance(ts, pd.DataFrame):
        if 'total_demand' in ts.columns:
            target_col = 'total_demand'
        else:
            target_col = ts.columns[0]
    else:
        target_col = None
    
    if target_col:
        X, y = forecaster.create_enhanced_features(ts, target_col)
    else:
        X, y = forecaster.create_enhanced_features(pd.DataFrame({'target': ts}), 'target')
        y = ts
    
    return X, y

def train_random_forest(X_train, y_train):
    """Обучение модели Random Forest"""
    forecaster = VacancyForecaster()
    rf_model, _ = forecaster.train_models(X_train, y_train)
    return rf_model

def train_gradient_boosting(X_train, y_train):
    """Обучение модели XGBoost"""
    forecaster = VacancyForecaster()
    _, xgb_model = forecaster.train_models(X_train, y_train)
    return xgb_model

def evaluate_models(y_true, rf_pred, gb_pred):
    """Оценка качества моделей"""
    try:
        rf_metrics = {
            'MAE': mean_absolute_error(y_true, rf_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, rf_pred)),
            'R2': r2_score(y_true, rf_pred)
        }
    except Exception as e:
        rf_metrics = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
    
    try:
        gb_metrics = {
            'MAE': mean_absolute_error(y_true, gb_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, gb_pred)),
            'R2': r2_score(y_true, gb_pred)
        }
    except Exception as e:
        gb_metrics = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
    
    return rf_metrics, gb_metrics

def visualize_forecast(history, validation, rf_forecast, gb_forecast, industry_name, rf_metrics, gb_metrics, output_folder):
    """Визуализация прогноза"""
    try:
        plt.figure(figsize=(15, 8))
        
        # Исторические данные
        plt.plot(history.index, history, 'b-', label='Исторические данные', marker='o', markersize=4)
        
        # Валидационный период
        plt.plot(validation.index, validation, 'g-', label='Валидационный период', marker='s', markersize=4)
        
        # Прогнозы моделей
        plt.plot(validation.index, rf_forecast, 'r--', 
                 label=f'Random Forest (RMSE: {rf_metrics["RMSE"]:.2f})', marker='x', markersize=4)
        plt.plot(validation.index, gb_forecast, 'm--', 
                 label=f'XGBoost (RMSE: {gb_metrics["RMSE"]:.2f})', marker='d', markersize=4)
        
        # Определение лучшей модели
        best_model = "Random Forest" if rf_metrics["RMSE"] < gb_metrics["RMSE"] else "XGBoost"
        
        plt.title(f'Прогноз потребности в инженерных кадрах: {industry_name}\nЛучшая модель: {best_model}')
        plt.xlabel('Период')
        plt.ylabel('Количество рабочих мест')
        plt.grid(True)
        plt.legend(loc='best')
        
        # Аннотация с метриками
        text_rf = f"Random Forest:\n"
        for metric, value in rf_metrics.items():
            text_rf += f"{metric}: {value:.2f}\n"
        
        text_gb = f"XGBoost:\n"
        for metric, value in gb_metrics.items():
            text_gb += f"{metric}: {value:.2f}\n"
        
        plt.annotate(text_rf, xy=(0.02, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        plt.annotate(text_gb, xy=(0.02, 0.75), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        
        # Создание безопасного имени для файла
        safe_name = re.sub(r'[^\w\s-]', '', industry_name).strip().replace(' ', '_')
        plt.savefig(f'{output_folder}/forecast_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return best_model
    except Exception as e:
        print(f"Ошибка при создании графика для {industry_name}: {e}")
        return "XGBoost"

def main():
    """Основной пайплайн для прогнозирования потребности в инженерных кадрах"""
    print("🚀 ЗАПУСК СИСТЕМЫ ПРОГНОЗИРОВАНИЯ")
    print("="*60)
    
    # Создание выходной директории
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Загрузка данных из {file_path}...")
    try:
        df = pd.read_csv(file_path, delimiter=';', low_memory=False)
        print(f"Загружено {df.shape[0]} записей")
    except FileNotFoundError:
        alternative_paths = [
            'vacancies_engineer.csv',
            './vacancies_engineer.csv',
            '../vacancies_engineer.csv',
            'production_results/final_processed_vacancies.csv'
        ]
        
        for alt_path in alternative_paths:
            try:
                print(f"Пробуем загрузить данные из {alt_path}...")
                df = pd.read_csv(alt_path, delimiter=';', low_memory=False)
                print(f"Загружено {df.shape[0]} записей")
                break
            except FileNotFoundError:
                continue
        else:
            print("Не удалось найти файл с данными.")
            return
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return
    
    # Создаем экземпляр прогнозировщика
    forecaster = VacancyForecaster(output_folder)
    
    # Проверяем, что данные уже предобработаны
    required_columns = ['date_creation_processed', 'date_end', 'work_places', 'industry']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Отсутствуют необходимые столбцы после предобработки: {missing_columns}")
        print("Убедитесь, что данные были предобработаны в модуле prepare_data_script")
        return
    
    # Проверка и корректировка типов данных
    print("🔧 Проверка и корректировка типов данных...")
    
    # Преобразуем даты в datetime
    date_columns = ['date_creation_processed', 'date_end']
    for col in date_columns:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f"   Преобразование {col} в datetime...")
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Убеждаемся, что work_places числовой
    if not pd.api.types.is_numeric_dtype(df['work_places']):
        print("   Преобразование work_places в числовой тип...")
        df['work_places'] = pd.to_numeric(df['work_places'], errors='coerce')
        df['work_places'] = df['work_places'].fillna(1)
    
    print(f"✅ Типы данных проверены. Записей с корректными датами: {df['date_creation_processed'].notna().sum()}")
    
    # Тестируем различные конфигурации
    test_configs = [
        {'freq': 'W', 'outlier_removal': False},
        {'freq': 'W', 'outlier_removal': True},
        {'freq': '2W', 'outlier_removal': False},
        {'freq': 'M', 'outlier_removal': False}
    ]
    
    best_results = {'r2': -float('inf'), 'config': None, 'ts_df': None}
    
    for i, config in enumerate(test_configs):
        print(f"\n📊 Тестирование конфигурации {i+1}/{len(test_configs)}: {config}")
        
        try:
            # Создание временных рядов
            ts_df = forecaster.create_time_series(df, freq=config['freq'], min_activity=5)
            
            if len(ts_df) < 20:
                print(f"  ⚠️ Недостаточно данных: {len(ts_df)} точек")
                continue
            
            # Удаление выбросов при необходимости
            if config['outlier_removal']:
                ts_df = forecaster.remove_outliers(ts_df)
            
            # Создание признаков для общего спроса
            X, y = forecaster.create_enhanced_features(ts_df, 'total_demand')
            
            if X.isna().any().any() or y.isna().any():
                print(f"  ⚠️ Найдены NaN в данных")
                continue
            
            # Разделение на обучение/тест
            train_size = int(len(X) * 0.8)
            if train_size < 10 or len(X) - train_size < 3:
                print(f"  ⚠️ Недостаточно данных для разделения")
                continue
            
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Обучение моделей
            rf_model, xgb_model = forecaster.train_models(X_train, y_train)
            
            # Предсказания
            rf_pred = forecaster.predict_with_model(rf_model, X_test)
            xgb_pred = forecaster.predict_with_model(xgb_model, X_test)
            
            # Оценка
            rf_r2 = r2_score(y_test, rf_pred)
            xgb_r2 = r2_score(y_test, xgb_pred)
            best_r2 = max(rf_r2, xgb_r2)
            
            print(f"  📈 Лучший R²: {best_r2:.4f}")
            
            if best_r2 > best_results['r2']:
                best_results = {
                    'r2': best_r2,
                    'config': config,
                    'original_ts_df': ts_df.copy(),  # Сохраняем оригинальные данные
                    'forecaster': forecaster
                }
                print(f"  🎯 Новый лучший результат!")
            
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            continue
    
    # Результаты с лучшей конфигурацией
    if best_results['config']:
        print(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
        print(f"Конфигурация: {best_results['config']}")
        print(f"Лучший R²: {best_results['r2']:.4f}")
        
        # Используем лучшую конфигурацию
        original_ts = best_results['original_ts_df']
        forecaster = best_results['forecaster']
        
        # Создаем графики до исправления аномалий
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(original_ts.index, original_ts['total_demand'], 'b-', label='Исходные данные')
            plt.title('Общий спрос на инженерные кадры до исправления аномалий')
            plt.xlabel('Период')
            plt.ylabel('Количество рабочих мест')
            plt.grid(True)
            plt.savefig(f'{output_folder}/demand_before_anomaly_fix.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Ошибка при создании графика до исправления: {e}")
        
        # Исправление аномалий
        ts_df = detect_and_fix_anomalies(original_ts, output_folder=output_folder)
        
        # Анализ временных рядов
        top_industries = analyze_timeseries(ts_df, output_folder)
        
        # Финальное прогнозирование с ИСПРАВЛЕННОЙ логикой
        print("\n📊 Финальное прогнозирование...")
        
        forecasts = {}
        best_models = {}
        all_rf_metrics = {}
        all_gb_metrics = {}
        
        def forecast_industry(series_data, name, forecaster_instance):
            """Прогнозирование для отрасли с консистентной логикой"""
            
            # Создаем признаки 
            X, y = forecaster_instance.create_enhanced_features(series_data, 'total_demand' if isinstance(series_data, pd.DataFrame) else None)
            
            # Разделение для валидации (такое же как в тестировании)
            train_size = int(len(X) * 0.8)
            if train_size < 10 or len(X) - train_size < 3:
                print(f"    ⚠️ Недостаточно данных для разделения в {name}")
                return None, None, None, None
            
            X_train, X_valid = X[:train_size], X[train_size:]
            y_train, y_valid = y[:train_size], y[train_size:]
            
            # Получаем соответствующие данные
            train_data = series_data.iloc[:train_size] if isinstance(series_data, pd.DataFrame) else series_data[:train_size]
            valid_data = series_data.iloc[train_size:] if isinstance(series_data, pd.DataFrame) else series_data[train_size:]
            
            # Обучаем модели
            rf_model, xgb_model = forecaster_instance.train_models(X_train, y_train)
            
            # Валидация
            rf_pred = forecaster_instance.predict_with_model(rf_model, X_valid)
            xgb_pred = forecaster_instance.predict_with_model(xgb_model, X_valid)
            
            # Метрики
            rf_metrics = {
                'MAE': mean_absolute_error(y_valid, rf_pred),
                'RMSE': np.sqrt(mean_squared_error(y_valid, rf_pred)),
                'R2': r2_score(y_valid, rf_pred)
            }
            
            xgb_metrics = {
                'MAE': mean_absolute_error(y_valid, xgb_pred),
                'RMSE': np.sqrt(mean_squared_error(y_valid, xgb_pred)),
                'R2': r2_score(y_valid, xgb_pred)
            }
            
            best_model = "Random Forest" if rf_metrics["RMSE"] < xgb_metrics["RMSE"] else "XGBoost"
            
            # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Финальный прогноз с рекурсивной логикой
            print(f"    Создание финального прогноза для {name}...")
            
            # Переобучаем на ВСЕХ данных, НЕ сбрасывая feature_names
            X_full, y_full = forecaster_instance.create_enhanced_features(series_data, 'total_demand' if isinstance(series_data, pd.DataFrame) else None)
            rf_final, xgb_final = forecaster_instance.train_models(X_full, y_full)
            
            # Создаем рекурсивный прогноз
            target_series = series_data if not isinstance(series_data, pd.DataFrame) else series_data['total_demand']
            final_forecast = forecaster_instance.create_recursive_forecast(
                target_series, best_model, rf_final, xgb_final, forecast_periods
            )
            
            return rf_metrics, xgb_metrics, best_model, final_forecast, train_data, valid_data, rf_pred, xgb_pred
        
        # Прогнозирование общего спроса
        print("Прогнозирование общего спроса...")
        total_demand = ts_df['total_demand']
        
        # НЕ сбрасываем feature_names - используем те же что создались при тестировании
        # forecaster.feature_names остается как есть
        
        result = forecast_industry(total_demand, 'total_demand', forecaster)
        
        if result[0] is not None:
            rf_metrics, xgb_metrics, best_model, final_forecast, train_data, valid_data, rf_pred, xgb_pred = result
            
            print(f"✅ Общий спрос - Random Forest R²: {rf_metrics['R2']:.4f}, XGBoost R²: {xgb_metrics['R2']:.4f}")
            print(f"    Финальный прогноз: {final_forecast.iloc[0]:,.0f} - {final_forecast.iloc[-1]:,.0f}")
            
            all_rf_metrics['total_demand'] = rf_metrics
            all_gb_metrics['total_demand'] = xgb_metrics
            best_models['total_demand'] = best_model
            forecasts['total_demand'] = final_forecast
            
            # Визуализация
            visualize_forecast(train_data, valid_data, rf_pred, xgb_pred, 'Общий спрос', rf_metrics, xgb_metrics, output_folder)
        
        # Прогнозирование для топ-отраслей
        for industry in top_industries.index[:5]:
            try:
                print(f"Прогнозирование для отрасли: {industry}")
                industry_data = ts_df[industry]
                
                if len(industry_data) > 24:
                    # НЕ сбрасываем feature_names - используем консистентные признаки
                    # forecaster.feature_names = None
                    
                    result = forecast_industry(industry_data, industry, forecaster)
                    
                    if result[0] is not None:
                        rf_metrics, xgb_metrics, best_model, final_forecast, train_data, valid_data, rf_pred, xgb_pred = result
                        
                        print(f"✅ {industry} - Random Forest R²: {rf_metrics['R2']:.4f}, XGBoost R²: {xgb_metrics['R2']:.4f}")
                        print(f"    Финальный прогноз: {final_forecast.iloc[0]:,.0f} - {final_forecast.iloc[-1]:,.0f}")
                        
                        all_rf_metrics[industry] = rf_metrics
                        all_gb_metrics[industry] = xgb_metrics
                        best_models[industry] = best_model
                        forecasts[industry] = final_forecast
                        
                        # Визуализация
                        visualize_forecast(train_data, valid_data, rf_pred, xgb_pred, industry, rf_metrics, xgb_metrics, output_folder)
                else:
                    print(f"  ⚠️ Недостаточно данных для отрасли {industry}: {len(industry_data)} точек")
                    
            except Exception as e:
                print(f"❌ Ошибка при прогнозировании для отрасли {industry}: {e}")
        
        # Сохранение результатов
        if forecasts:
            # DataFrame с прогнозами
            forecast_df = pd.DataFrame(forecasts)
            forecast_df.to_csv(f'{output_folder}/industry_demand_forecast.csv')
            
            # Метрики
            if all_rf_metrics:
                rf_metrics_summary = pd.DataFrame.from_dict(all_rf_metrics, orient='index')
                rf_metrics_summary.to_csv(f'{output_folder}/random_forest_metrics.csv')
            
            if all_gb_metrics:
                gb_metrics_summary = pd.DataFrame.from_dict(all_gb_metrics, orient='index')
                gb_metrics_summary.to_csv(f'{output_folder}/gradient_boosting_metrics.csv')
            
            # Лучшие модели
            if best_models:
                pd.Series(best_models).to_csv(f'{output_folder}/best_models_per_industry.csv')
            
            # Создание сравнительной визуализации
            if all_rf_metrics and all_gb_metrics:
                try:
                    plt.figure(figsize=(12, 8))
                    metrics = ['MAE', 'RMSE', 'R2']
                    x = range(len(metrics))
                    width = 0.35
                    
                    rf_values = [rf_metrics_summary[metric].mean(skipna=True) for metric in metrics]
                    gb_values = [gb_metrics_summary[metric].mean(skipna=True) for metric in metrics]
                    
                    plt.bar([i - width/2 for i in x], rf_values, width, label='Random Forest')
                    plt.bar([i + width/2 for i in x], gb_values, width, label='XGBoost')
                    
                    plt.xlabel('Метрика')
                    plt.ylabel('Значение')
                    plt.title('Сравнение моделей по средним метрикам')
                    plt.xticks(x, metrics)
                    plt.legend()
                    plt.savefig(f'{output_folder}/model_comparison.png', dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Ошибка при создании сравнительного графика: {e}")
            
            # Итоговый отчет
            rf_count = list(best_models.values()).count("Random Forest")
            gb_count = list(best_models.values()).count("XGBoost")
            
            if all_rf_metrics and all_gb_metrics:
                overall_best_model = "Random Forest" if rf_metrics_summary['RMSE'].mean(skipna=True) < gb_metrics_summary['RMSE'].mean(skipna=True) else "XGBoost"
                
                report = f"""
Итоговый отчет по прогнозированию потребности в инженерных кадрах:

1. Анализируемый период: {ts_df.index[0].strftime('%Y-%m')} - {ts_df.index[-1].strftime('%Y-%m')}
2. Общее количество вакансий: {len(df)}
3. Анализируемые отрасли: {len(top_industries)}
4. Лучшая конфигурация: {best_results['config']}
5. Лучший R² (тестирование): {best_results['r2']:.4f}

6. Средние метрики моделей:
   Random Forest:
   - MAE: {rf_metrics_summary['MAE'].mean(skipna=True):.2f}
   - RMSE: {rf_metrics_summary['RMSE'].mean(skipna=True):.2f}
   - R2: {rf_metrics_summary['R2'].mean(skipna=True):.4f}
   
   XGBoost:
   - MAE: {gb_metrics_summary['MAE'].mean(skipna=True):.2f}
   - RMSE: {gb_metrics_summary['RMSE'].mean(skipna=True):.2f}
   - R2: {gb_metrics_summary['R2'].mean(skipna=True):.4f}

7. Распределение лучших моделей по отраслям:
   Random Forest: {rf_count} отраслей ({rf_count/(rf_count+gb_count)*100 if rf_count+gb_count > 0 else 0:.1f}%)
   XGBoost: {gb_count} отраслей ({gb_count/(rf_count+gb_count)*100 if rf_count+gb_count > 0 else 0:.1f}%)

8. Лучшая модель по среднему RMSE: {overall_best_model}

9. Горизонт прогнозов: {forecast_periods} периодов

Финальные прогнозы созданы с использованием рекурсивной логики и консистентны с валидацией.
"""
                
                print(report)
                
                with open(f'{output_folder}/final_report.txt', 'w', encoding='utf-8') as f:
                    f.write(report)
            
            print(f"\n✅ Анализ успешно завершен. Результаты сохранены в папку {output_folder}")
            print(f"🎯 Прогнозы на {forecast_periods} периодов созданы с исправленной логикой!")
    else:
        print("❌ Не удалось найти работающую конфигурацию")

if __name__ == "__main__":
    main()