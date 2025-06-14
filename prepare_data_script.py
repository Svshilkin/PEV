import pandas as pd
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import os
import warnings
from scipy import stats

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil не установлен. Для оптимального управления памяти рекомендуется установить: pip install psutil")

warnings.filterwarnings('ignore')

def from_raw(file, chunk_size=50000):
    """Обработка сырых данных для извлечения инженерных вакансий с оптимизацией памяти"""
    cooked = 'vacancies_cooked.csv'

    # Столбцы для удаления (избыточные или неинформативные)
    columns_to_remove = {
        'driver_licence_e', 'accommodation_housing', 'disabled', 'inner_info_metro_ids',
        'retraining_grant', 'date_time_change_inner_info', 'payment_meals',
        'retraining_capability', 'caring_workers', 'federal_district',
        'payment_sports_activities', 'job_benefits', 'retraining_condition',
        'additional_info', 'metro_station', 'incentive_compensation_transport_compensation',
        'premium_size', 'driver_licence_c', 'driver_licence_b', 'premium_type',
        'career_perspective', 'workers_with_disabled_children', 'dms', 'released_persons',
        'driver_licence_d', 'drive_licences', 'social_protecteds_social_protected',
        'single_parent', 'vouchers_health_institutions', 'retraining_grant_value',
        'requirements_id_priority_category', 'need_medcard', 'driver_licence_a',
        'vac_url', 'accommodation_capability', 'is_uzbekistan_recruitment'
    }

    print("🔧 Обработка сырых данных (оптимизированный режим)...")
    engineer_count = 0
    total_count = 0
    chunk_count = 0
    
    # Определяем разделитель CSV файла
    delimiter = ';'  # Используем точку с запятой как разделитель
    
    # Получаем заголовки из первой строки
    with open(file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=delimiter)
        header = next(reader)

    print(f"📋 Найдено столбцов в исходном файле: {len(header)}")
    print(f"📋 Первые 10 столбцов: {header[:10]}")
    
    # Проверяем наличие столбца title
    if 'title' not in header:
        print(f"❌ Столбец 'title' не найден!")
        print(f"📋 Все столбцы в файле: {header}")
        # Попробуем найти похожие столбцы
        title_like = [col for col in header if 'title' in col.lower() or 'название' in col.lower() or 'наименование' in col.lower()]
        if title_like:
            print(f"🔍 Найдены похожие столбцы: {title_like}")
        raise SystemExit('❌ Столбец "title" не найден в данных!')

    # Определяем индексы столбцов для сохранения
    keep_indices = [i for i, col in enumerate(header) if col not in columns_to_remove]
    new_header = [header[i] for i in keep_indices]

    print(f"📋 Столбцов после фильтрации: {len(new_header)}")

    # Находим индекс столбца "title"
    try:
        title_index = new_header.index('title')
        print(f"✅ Столбец 'title' найден на позиции: {title_index}")
    except ValueError:
        print(f"❌ Столбец 'title' отсутствует после фильтрации!")
        print(f"📋 Столбцы после фильтрации: {new_header}")
        raise SystemExit('❌ Столбец "title" не найден в отфильтрованных данных!')

    # Создаем выходной файл и записываем заголовок
    with open(cooked, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter=';')
        writer.writerow(new_header)

        # Обрабатываем файл по частям
        with open(file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter=delimiter)
            next(reader)  # Пропускаем заголовок
            
            chunk_data = []
            for row in reader:
                total_count += 1
                chunk_data.append(row)
                
                # Обрабатываем чанк когда он достигнет нужного размера
                if len(chunk_data) >= chunk_size:
                    chunk_count += 1
                    engineer_found = process_raw_chunk(chunk_data, keep_indices, title_index, writer)
                    engineer_count += engineer_found
                    chunk_data = []

                    
                    # Принудительная очистка памяти
                    import gc
                    gc.collect()
            
            # Обрабатываем оставшиеся данные
            if chunk_data:
                chunk_count += 1
                engineer_found = process_raw_chunk(chunk_data, keep_indices, title_index, writer)
                engineer_count += engineer_found
                print(f"   Обработан финальный чанк {chunk_count}: {total_count:,} строк, найдено инженерных: {engineer_count:,}")

    print(f"✅ Обработано {total_count:,} записей")
    print(f"📊 Найдено {engineer_count:,} инженерных вакансий ({engineer_count/total_count*100:.1f}%)")
    
    return cooked

def process_raw_chunk(chunk_data, keep_indices, title_index, writer):
    """Обработка чанка сырых данных"""
    engineer_count = 0
    
    for row in chunk_data:
        new_row = [row[i] if i < len(row) else '' for i in keep_indices]
        
        # Проверяем вхождение "инженер" без учета регистра
        if title_index < len(new_row) and 'инженер' in new_row[title_index].lower():
            writer.writerow(new_row)
            engineer_count += 1
    
    return engineer_count

def normalize_workplaces(value):
    """Нормализация количества рабочих мест"""
    if pd.isna(value):
        return 1

    try:
        # Обработка строковых значений вида "от X до Y"
        if isinstance(value, str) and 'до' in value:
            match = re.search(r'от\s+(\d+)\s+до\s+(\d+)', value)
            if match:
                min_val = int(match.group(1))
                max_val = int(match.group(2))
                return (min_val + max_val) // 2
        elif isinstance(value, str) and 'от' in value:
            match = re.search(r'от\s+(\d+)', value)
            if match:
                return int(match.group(1))
    except:
        pass

    # Преобразование в целое число с проверкой
    try:
        result = max(1, int(float(value)))
        # Ограничиваем максимальное значение для предотвращения выбросов
        return min(result, 10000)  
    except:
        return 1

def encode_category_by_target(df, feature, target='work_places', drop_original=False):
    """
    Кодирование категориальных признаков средним значением целевой переменной.
    """
    if feature not in df.columns:
        print(f"⚠️ Признак {feature} отсутствует в данных")
        return df

    # Заполняем пропуски
    df[feature] = df[feature].fillna('Unknown')

    # Создаем словарь для замены с учетом регуляризации
    category_stats = df.groupby(feature)[target].agg(['mean', 'count']).reset_index()
    global_mean = df[target].mean()
    
    # Применяем сглаживание для категорий с малым количеством наблюдений
    min_samples = 10
    encoding_dict = {}
    
    for _, row in category_stats.iterrows():
        category = row[feature]
        category_mean = row['mean']
        category_count = row['count']
        
        if category_count < min_samples:
            # Сглаживание по Лапласу
            alpha = min_samples
            smoothed_mean = (category_mean * category_count + global_mean * alpha) / (category_count + alpha)
            encoding_dict[category] = smoothed_mean
        else:
            encoding_dict[category] = category_mean

    # Создаем новый признак с кодированием
    df[f'{feature}_encoded'] = df[feature].map(lambda x: encoding_dict.get(x, global_mean))

    # Удаляем оригинальный признак, если нужно
    if drop_original and feature != target:
        df = df.drop(feature, axis=1)

    return df

def discretize_numeric_feature(df, feature, num_bins=10, target='work_places', drop_original=False):
    """
    Дискретизация числовых признаков с адаптивными бинами.
    """
    if feature not in df.columns:
        print(f"⚠️ Признак {feature} отсутствует в данных")
        return df

    try:
        # Преобразуем значения в числовые
        if not pd.api.types.is_numeric_dtype(df[feature]):
            df[feature] = pd.to_numeric(df[feature], errors='coerce')

        # Удаляем выбросы перед дискретизацией
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Ограничиваем выбросы
        df[feature] = df[feature].clip(lower_bound, upper_bound)
        
        # Заполняем пропуски медианой
        df[feature] = df[feature].fillna(df[feature].median())

        # Используем квантильную дискретизацию для более равномерного распределения
        try:
            df[f'{feature}_bin_idx'], bin_edges = pd.qcut(df[feature], q=num_bins, 
                                                         labels=False, duplicates='drop', retbins=True)
        except ValueError:
            # Если квантильная дискретизация не работает, используем равномерные интервалы
            df[f'{feature}_bin_idx'], bin_edges = pd.cut(df[feature], bins=num_bins, 
                                                        labels=False, retbins=True)

        # Рассчитываем среднее значение целевой переменной для каждого бина
        bin_means = {}
        global_mean = df[target].mean()
        
        for bin_idx in range(len(bin_edges) - 1):
            mask = df[f'{feature}_bin_idx'] == bin_idx
            if mask.any() and mask.sum() > 0:
                bin_means[bin_idx] = df.loc[mask, target].mean()
            else:
                bin_means[bin_idx] = global_mean

        # Создаем закодированный признак
        df[f'{feature}_encoded'] = df[f'{feature}_bin_idx'].map(bin_means)
        df[f'{feature}_encoded'] = df[f'{feature}_encoded'].fillna(global_mean)

        # Удаляем промежуточный столбец
        df = df.drop(f'{feature}_bin_idx', axis=1)

        # Удаляем оригинальный признак, если нужно
        if drop_original and feature != target:
            df = df.drop(feature, axis=1)

    except Exception as e:
        print(f"⚠️ Ошибка при обработке признака {feature}: {e}")
        # В случае ошибки просто заполняем средним значением
        df[f'{feature}_encoded'] = df[target].mean()

        if drop_original and feature != target:
            df = df.drop(feature, axis=1)

    return df

def process_date_feature(df, date_col, mistake_col=None, drop_original=False):
    """
    Обработка признаков дат.
    """
    if date_col not in df.columns:
        print(f"⚠️ Признак {date_col} отсутствует в данных")
        return df

    # Копируем признак для обработки
    df[f'{date_col}_processed'] = df[date_col].copy()

    # Проверяем наличие ошибок
    if mistake_col and mistake_col in df.columns:
        df.loc[df[mistake_col] == '1', f'{date_col}_processed'] = None

    # Преобразуем в datetime
    df[f'{date_col}_processed'] = pd.to_datetime(df[f'{date_col}_processed'], errors='coerce')

    # Если преобразование прошло успешно, добавляем временные признаки
    if not df[f'{date_col}_processed'].isna().all():
        # Основные временные признаки
        df[f'{date_col}_year'] = df[f'{date_col}_processed'].dt.year
        df[f'{date_col}_month'] = df[f'{date_col}_processed'].dt.month
        df[f'{date_col}_quarter'] = df[f'{date_col}_processed'].dt.quarter
        df[f'{date_col}_day_of_week'] = df[f'{date_col}_processed'].dt.dayofweek
        df[f'{date_col}_day_of_year'] = df[f'{date_col}_processed'].dt.dayofyear

        # Кодируем временные признаки
        for temp_feature in [f'{date_col}_year', f'{date_col}_month', f'{date_col}_quarter']:
            df = encode_category_by_target(df, temp_feature, drop_original=False)

    # Удаляем оригинальные признаки, если нужно
    if drop_original:
        if date_col in df.columns:
            df = df.drop(date_col, axis=1)
        if mistake_col and mistake_col in df.columns:
            df = df.drop(mistake_col, axis=1)

    return df

def process_chunk(chunk):
    """
    Обработка одного чанка данных с полной логикой предобработки
    """
    try:
        # 1. Нормализация количества рабочих мест
        chunk['work_places'] = chunk['work_places'].apply(normalize_workplaces)
        
        # 2. Обработка дат
        date_columns = [
            ('date_creation', 'date_creation_mistake'),
            ('date_posted', 'date_posted_mistake'),
            ('date_inactivation', None),
            ('date_last_updated', None),
            ('date_modify_inner_info', 'date_modify_inner_info_mistake')
        ]

        for date_col, mistake_col in date_columns:
            if date_col in chunk.columns:
                try:
                    chunk = process_date_feature(chunk, date_col, mistake_col, drop_original=False)
                except Exception as e:
                    print(f"      ⚠️ Ошибка при обработке даты {date_col}: {e}")
                    continue

        # 3. Обработка категориальных признаков
        # Специальная обработка отрасли
        if 'industry' in chunk.columns:
            try:
                chunk['industry'] = chunk['industry'].fillna('Unknown')
                chunk = encode_category_by_target(chunk, 'industry', drop_original=False)
            except Exception as e:
                print(f"      ⚠️ Ошибка при обработке отрасли: {e}")

        categorical_features = [
            'employment_type', 'education_requirements_education_type',
            'region', 'profession', 'organization'
        ]

        for feature in categorical_features:
            if feature in chunk.columns:
                try:
                    chunk = encode_category_by_target(chunk, feature, drop_original=False)
                except Exception as e:
                    print(f"      ⚠️ Ошибка при обработке {feature}: {e}")
                    continue

        # 4. Обработка числовых признаков
        numeric_features = [
            'base_salary_min', 'base_salary_max', 'experience_requirements'
        ]

        for feature in numeric_features:
            if feature in chunk.columns:
                try:
                    chunk = discretize_numeric_feature(chunk, feature, drop_original=False)
                except Exception as e:
                    print(f"      ⚠️ Ошибка при обработке {feature}: {e}")
                    continue

        # 5. КЛЮЧЕВАЯ ЛОГИКА: Создание date_end и vacancy_duration (перенесено из improved_solution.py)
        try:
            # Определение конца активности вакансии
            if 'date_inactivation_processed' in chunk.columns and 'date_last_updated_processed' in chunk.columns:
                chunk['date_end'] = chunk['date_inactivation_processed'].fillna(chunk['date_last_updated_processed'])
            elif 'date_last_updated_processed' in chunk.columns:
                chunk['date_end'] = chunk['date_last_updated_processed']
            elif 'date_inactivation_processed' in chunk.columns:
                chunk['date_end'] = chunk['date_inactivation_processed']
            else:
                # Если нет данных об окончании, используем среднюю продолжительность
                if 'date_creation_processed' in chunk.columns:
                    chunk['date_end'] = chunk['date_creation_processed'] + pd.Timedelta(days=77)

            # Заполняем пропуски в date_end
            if 'date_end' in chunk.columns and 'date_creation_processed' in chunk.columns:
                mask_no_end = chunk['date_end'].isna()
                if mask_no_end.any():
                    chunk.loc[mask_no_end, 'date_end'] = chunk.loc[mask_no_end, 'date_creation_processed'] + pd.Timedelta(days=77)

                # Длительность активности
                chunk['vacancy_duration'] = (chunk['date_end'] - chunk['date_creation_processed']).dt.days
                chunk['vacancy_duration'] = chunk['vacancy_duration'].fillna(77).clip(lower=1, upper=365)
                
        except Exception as e:
            print(f"      ⚠️ Ошибка при создании date_end: {e}")

        # 6. Фильтрация данных (логика из improved версии)
        initial_size = len(chunk)
        
        # Убираем записи без даты создания
        if 'date_creation_processed' in chunk.columns:
            chunk = chunk[chunk['date_creation_processed'].notna()]
        
        # КРИТИЧЕСКИ ВАЖНО: Проверка логичности дат (из improved_solution.py)
        if 'date_end' in chunk.columns and 'date_creation_processed' in chunk.columns:
            chunk = chunk[chunk['date_end'] >= chunk['date_creation_processed']]
        
        # Убираем записи с неизвестной отраслью (если есть поле industry)
        if 'industry' in chunk.columns:
            chunk = chunk[chunk['industry'] != 'Unknown']
        
        # Фильтрация по временному диапазону (основной период анализа)
        if 'date_creation_processed' in chunk.columns:
            chunk = chunk[(chunk['date_creation_processed'] >= '2018-08-01') & 
                         (chunk['date_creation_processed'] <= '2021-07-31')]

        # Убеждаемся что типы данных корректны
        if 'date_creation_processed' in chunk.columns:
            chunk['date_creation_processed'] = pd.to_datetime(chunk['date_creation_processed'], errors='coerce')
        if 'date_end' in chunk.columns:
            chunk['date_end'] = pd.to_datetime(chunk['date_end'], errors='coerce')
        if 'work_places' in chunk.columns:
            chunk['work_places'] = pd.to_numeric(chunk['work_places'], errors='coerce').fillna(1)

        return chunk
        
    except Exception as e:
        print(f"      ❌ Критическая ошибка при обработке чанка: {e}")
        # Возвращаем пустой DataFrame с теми же столбцами
        return pd.DataFrame(columns=chunk.columns if 'chunk' in locals() else [])

def create_visualizations(df, output_dir):
    """Создание визуализаций для анализа данных"""
    print("📊 Создание визуализаций...")
    
    plt.style.use('seaborn-v0_8')
    
    # 1. Распределение количества рабочих мест с логарифмической шкалой
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(df['work_places'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Распределение количества рабочих мест')
    plt.xlabel('Количество рабочих мест')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(np.log1p(df['work_places']), bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Распределение log(1 + работных мест)')
    plt.xlabel('log(1 + количество рабочих мест)')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)
    
    # 3. Топ-15 отраслей по количеству вакансий
    plt.subplot(2, 2, 3)
    if 'industry' in df.columns:
        top_industries = df['industry'].value_counts().head(15)
        plt.barh(range(len(top_industries)), top_industries.values)
        plt.yticks(range(len(top_industries)), [str(ind)[:20] + '...' if len(str(ind)) > 20 else str(ind) 
                                               for ind in top_industries.index])
        plt.title('Топ-15 отраслей по количеству вакансий')
        plt.xlabel('Количество вакансий')
        plt.grid(True, alpha=0.3)
    
    # 4. Временная динамика создания вакансий
    plt.subplot(2, 2, 4)
    if 'date_creation_processed' in df.columns:
        monthly_counts = df.set_index('date_creation_processed').resample('M').size()
        plt.plot(monthly_counts.index, monthly_counts.values, marker='o')
        plt.title('Динамика создания вакансий по месяцам')
        plt.xlabel('Дата')
        plt.ylabel('Количество вакансий')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Анализ отраслей по средней потребности в кадрах
    if 'industry' in df.columns:
        plt.figure(figsize=(15, 8))
        
        industry_stats = df.groupby('industry').agg({
            'work_places': ['sum', 'mean', 'count']
        }).round(2)
        
        industry_stats.columns = ['Общий_спрос', 'Средний_спрос', 'Количество_вакансий']
        industry_stats = industry_stats.sort_values('Общий_спрос', ascending=False).head(20)
        
        plt.subplot(1, 2, 1)
        plt.barh(range(len(industry_stats)), industry_stats['Общий_спрос'])
        plt.yticks(range(len(industry_stats)), 
                  [str(ind)[:25] + '...' if len(str(ind)) > 25 else str(ind) 
                   for ind in industry_stats.index])
        plt.title('Топ-20 отраслей по общему спросу на кадры')
        plt.xlabel('Общий спрос (рабочих мест)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Фильтруем отрасли с минимальным количеством вакансий для стабильности оценки
        stable_industries = industry_stats[industry_stats['Количество_вакансий'] >= 10]
        stable_industries = stable_industries.sort_values('Средний_спрос', ascending=False).head(15)
        
        plt.barh(range(len(stable_industries)), stable_industries['Средний_спрос'], color='orange')
        plt.yticks(range(len(stable_industries)), 
                  [str(ind)[:25] + '...' if len(str(ind)) > 25 else str(ind) 
                   for ind in stable_industries.index])
        plt.title('Топ-15 отраслей по среднему спросу\n(мин. 10 вакансий)')
        plt.xlabel('Средний спрос (рабочих мест на вакансию)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/industry_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Анализ заработных плат (если данные есть)
    salary_columns = ['base_salary_min', 'base_salary_max']
    available_salary_cols = [col for col in salary_columns if col in df.columns]
    
    if available_salary_cols:
        plt.figure(figsize=(12, 6))
        
        for i, col in enumerate(available_salary_cols):
            plt.subplot(1, len(available_salary_cols), i+1)
            
            # Удаляем выбросы для визуализации
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            filtered_data = df[col][(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
            
            plt.hist(filtered_data.dropna(), bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'Распределение {col}')
            plt.xlabel('Зарплата (руб.)')
            plt.ylabel('Частота')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/salary_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✅ Визуализации созданы")

def prepare_vacancies_data(raw, input_file, output_file=None, visualize=True, chunk_size=25000):
    """
    Подготовка данных о вакансиях для анализа с оптимизацией памяти.
    Включает всю логику предобработки из improved версии.
    """
    print("🚀 НАЧАЛО ПОДГОТОВКИ ДАННЫХ")
    print("="*50)
    
    try:
        # Обработка сырых данных, если нужно
        if raw:
            input_file = from_raw(input_file, chunk_size=chunk_size//2)

        # Оптимизированная загрузка данных по частям для экономии памяти
        print(f"📂 Загрузка данных из {input_file} (оптимизированный режим)...")
        
        # Определяем размер чанка на основе доступной памяти
        if PSUTIL_AVAILABLE:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Более консервативные размеры для 20ГБ файла на 40ГБ RAM
            if available_memory_gb > 30:
                chunk_size = min(chunk_size, 15000)  # Ограничиваем даже для большой памяти
            elif available_memory_gb > 20:
                chunk_size = min(chunk_size, 10000)   
            elif available_memory_gb > 15:
                chunk_size = min(chunk_size, 7500)   
            elif available_memory_gb > 10:
                chunk_size = min(chunk_size, 5000)   
            else:
                chunk_size = min(chunk_size, 2500)   # Очень маленькие чанки для ограниченной памяти
            
            print(f"📊 Доступно памяти: {available_memory_gb:.1f} ГБ")
            print(f"🔧 Размер чанка: {chunk_size:,} строк")
        else:
            chunk_size = min(chunk_size, 10000)  # Консервативное значение по умолчанию
            print(f"🔧 Размер чанка по умолчанию: {chunk_size:,} строк")
        
        # Читаем данные по частям и обрабатываем
        processed_chunks = []
        total_rows = 0
        chunk_count = 0
        
        print(f"📊 Начинаем чанковую обработку (размер чанка: {chunk_size:,} строк)...")
        
        try:
            for chunk in pd.read_csv(input_file, delimiter=';', chunksize=chunk_size, 
                                    low_memory=False, on_bad_lines='skip'):
                chunk_count += 1
                current_chunk_size = len(chunk)
                total_rows += current_chunk_size

                
                try:
                    # Применяем полную обработку к чанку
                    processed_chunk = process_chunk(chunk)
                    
                    if len(processed_chunk) > 0:
                        processed_chunks.append(processed_chunk)
                    else:
                        print(f"      ⚠️ Чанк отфильтрован полностью")
                    
                    # Принудительная очистка памяти
                    del chunk, processed_chunk
                    import gc
                    gc.collect()
                    
                except Exception as chunk_error:
                    print(f"   ⚠️ Ошибка при обработке чанка {chunk_count}: {chunk_error}")
                    print(f"   Пропускаем этот чанк и продолжаем...")
                    continue
                    
        except Exception as read_error:
            print(f"❌ Ошибка при чтении файла: {read_error}")
            return None
        
        # Объединяем все обработанные чанки
        print(f"🔄 Объединение {len(processed_chunks)} обработанных чанков...")
        if not processed_chunks:
            print("❌ Нет данных для обработки после фильтрации")
            return None
            
        df = pd.concat(processed_chunks, ignore_index=True)
        
        # Освобождаем память от чанков
        del processed_chunks
        import gc
        gc.collect()
        
        original_shape = (total_rows, len(df.columns))
        print(f"✅ Обработано {total_rows:,} строк, итоговый размер: {df.shape[0]:,} × {df.shape[1]}")

        # Финальная статистика после чанковой обработки
        filtered_count = len(df)
        retention_rate = (filtered_count / total_rows) * 100
        print(f"📈 Сохранено {retention_rate:.1f}% записей после фильтрации")

        # Статистика по work_places
        print(f"   📊 Статистика work_places:")
        print(f"      Среднее: {df['work_places'].mean():.2f}")
        print(f"      Медиана: {df['work_places'].median():.2f}")
        print(f"      Макс: {df['work_places'].max():,}")
        print(f"      99-й процентиль: {df['work_places'].quantile(0.99):.0f}")

        # Создание визуализаций
        if visualize:
            output_dir = os.path.dirname(output_file) if output_file else '.'
            os.makedirs(output_dir, exist_ok=True)
            create_visualizations(df, output_dir)

        # Итоговая статистика
        print("\n📊 ИТОГОВАЯ СТАТИСТИКА:")
        print(f"   Исходный размер: {original_shape[0]:,} × {original_shape[1]}")
        print(f"   Итоговый размер: {df.shape[0]:,} × {df.shape[1]}")
        print(f"   Закодированных признаков: {len([col for col in df.columns if '_encoded' in col])}")
        
        if 'industry' in df.columns:
            print(f"   Уникальных отраслей: {df['industry'].nunique()}")
            print(f"   Топ-3 отрасли: {list(df['industry'].value_counts().head(3).index)}")
        
        if 'date_creation_processed' in df.columns:
            print(f"   Временной диапазон: {df['date_creation_processed'].min().strftime('%Y-%m-%d')} - {df['date_creation_processed'].max().strftime('%Y-%m-%d')}")
        print(f"   Общий спрос (рабочих мест): {df['work_places'].sum():,}")

        # Сохраняем подготовленные данные
        if output_file:
            print(f"💾 Сохранение подготовленных данных...")
            df.to_csv(output_file, index=False, encoding='utf-8', sep=';')
            print(f"💾 Подготовленные данные сохранены: {output_file}")

        # Очищаем временный файл, если создавался
        if raw and os.path.exists(input_file) and input_file != 'vacancies_engineer.csv':
            os.remove(input_file)

        print("✅ ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА УСПЕШНО")
        return df

    except Exception as e:
        print(f"❌ Ошибка при подготовке данных: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_categorical_feature(df, feature):
    """
    Обрабатывает категориальные признаки (совместимость со старым кодом).
    """
    return encode_category_by_target(df, feature, drop_original=False)


if __name__ == "__main__":
    # Пути к файлам
    input_file = "vacancies.csv"
    output_file = "results/processed_vacancies_engineer.csv"

    print("🎯 ЗАПУСК ПРЕДОБРАБОТКИ ДАННЫХ")
    print("="*60)

    # Определяем размер чанка на основе доступной памяти
    if PSUTIL_AVAILABLE:
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Очень консервативные размеры для 20ГБ данных
        if available_memory_gb > 30:
            chunk_size = 10000   
        elif available_memory_gb > 20:
            chunk_size = 7500   
        elif available_memory_gb > 15:
            chunk_size = 5000   
        elif available_memory_gb > 10:
            chunk_size = 3000   
        else:
            chunk_size = 1500   
        
        print(f"📊 Доступно памяти: {available_memory_gb:.1f} ГБ")
        print(f"🔧 Размер чанка: {chunk_size:,} строк")
    else:
        chunk_size = 5000  # Очень консервативное значение по умолчанию
        print(f"🔧 Размер чанка по умолчанию: {chunk_size:,} строк")

    # Подготовка данных
    processed_df = prepare_vacancies_data(
        raw=True,
        input_file=input_file, 
        output_file=output_file, 
        visualize=False,
        chunk_size=chunk_size
    )

    if processed_df is not None:
        print("\n🔍 ПРИМЕРЫ ПОДГОТОВЛЕННЫХ ДАННЫХ:")
        print("="*50)
        print(processed_df.head())

        print("\n📊 ИНФОРМАЦИЯ О ЗАКОДИРОВАННЫХ ПРИЗНАКАХ:")
        print("="*50)
        encoded_columns = [col for col in processed_df.columns if '_encoded' in col]
        
        for i, col in enumerate(encoded_columns[:10]):  # Показываем первые 10
            print(f"   {i+1:2d}. {col:30s}: {processed_df[col].min():7.2f} - {processed_df[col].max():7.2f} "
                  f"(среднее: {processed_df[col].mean():7.2f})")
        
        if len(encoded_columns) > 10:
            print(f"   ... и еще {len(encoded_columns) - 10} закодированных признаков")

        print(f"\n📈 СВОДКА:")
        print("="*50)
        print(f"   ✅ Всего признаков: {len(processed_df.columns)}")
        print(f"   🔢 Закодированных: {len(encoded_columns)}")
        print(f"   📊 Записей: {len(processed_df):,}")
        print(f"   🏭 Отраслей: {processed_df['industry'].nunique() if 'industry' in processed_df.columns else 'N/A'}")
        print(f"   💼 Общий спрос: {processed_df['work_places'].sum():,} рабочих мест")
        
        print(f"\n✅ Данные готовы для использования в системе прогнозирования!")
        print(f"📂 Файл сохранен: {output_file}")
    else:
        print("\n❌ Ошибка при обработке данных. Проверьте входной файл и повторите попытку.")
