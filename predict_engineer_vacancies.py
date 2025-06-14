import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Импортируем функции из наших модулей
from prepare_data_script import prepare_vacancies_data
from solution import main as run_forecasting

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Прогнозирование потребности в инженерных кадрах')
    
    parser.add_argument('--input', type=str, default='vacancies.csv',
                        help='Путь к входному CSV-файлу с вакансиями')
    
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Директория для сохранения результатов')
    
    parser.add_argument('--skip-prepare', action='store_true',
                        help='Пропустить этап подготовки данных')
    
    parser.add_argument('--processed-data', type=str, default=None,
                        help='Путь к уже подготовленным данным (для использования с --skip-prepare)')

    parser.add_argument('--is-raw', action='store_true',
                        help='Используются ли сырые данные')
    
    parser.add_argument('--forecast-periods', type=int, default=6,
                        help='Количество периодов для прогнозирования')
    
    parser.add_argument('--validation-periods', type=int, default=6,
                        help='Количество периодов для валидации')
    
    parser.add_argument('--auto-anomaly-detection', action='store_true', default=True,
                        help='Автоматическое обнаружение и исправление аномалий (по умолчанию включено)')
    
    parser.add_argument('--anomaly-threshold', type=float, default=3.0,
                        help='Порог для обнаружения аномалий (Z-score)')
    
    parser.add_argument('--auto-config', action='store_true',
                        help='Автоматический выбор лучшей конфигурации')
    
    return parser.parse_args()

def run_pipeline():
    """Запуск полного пайплайна прогнозирования"""
    print("🚀 СИСТЕМА ПРОГНОЗИРОВАНИЯ ИНЖЕНЕРНЫХ КАДРОВ")
    print("="*70)
    
    # Получаем аргументы
    args = parse_arguments()
    
    # Создаем выходную директорию, если она не существует
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Определяем пути к файлам
    input_file = args.input
    processed_file = args.processed_data if args.processed_data else os.path.join(args.output_dir, 'processed_vacancies.csv')
    
    # Шаг 1: Подготовка данных
    if not args.skip_prepare:
        print("\n" + "=" * 80)
        print("ЭТАП 1: ПОДГОТОВКА ДАННЫХ")
        print("=" * 80)
        
        # Запускаем подготовку данных
        processed_df = prepare_vacancies_data(
            raw=args.is_raw,
            input_file=input_file,
            output_file=processed_file,
            visualize=True
        )
        
        if processed_df is None:
            print("\nОшибка при подготовке данных. Завершение работы.")
            return
    else:
        print("\nПропускаем этап подготовки данных, используя уже подготовленные данные.")
        
        # Проверяем, существует ли файл с подготовленными данными
        if not os.path.exists(processed_file):
            print(f"Ошибка: файл с подготовленными данными не найден: {processed_file}")
            print("Пожалуйста, укажите правильный путь или уберите флаг --skip-prepare")
            return
    
    # Шаг 2: Запуск прогнозирования
    print("\n" + "=" * 80)
    print("ЭТАП 2: ПРОГНОЗИРОВАНИЕ")
    print("=" * 80)
    
    # Переопределяем необходимые глобальные переменные для прогнозирования
    import solution
    solution.file_path = processed_file
    solution.output_folder = args.output_dir
    solution.forecast_periods = args.forecast_periods
    solution.validation_periods = args.validation_periods
    
    # Информация о параметрах
    if args.auto_config:
        print("🔧 Включен режим автоматического выбора конфигурации")
        print("   Система протестирует различные параметры и выберет лучшие")
    
    print(f"📊 Параметры прогнозирования:")
    print(f"   • Горизонт прогноза: {args.forecast_periods} периодов")
    print(f"   • Валидационный период: {args.validation_periods} периодов")
    print(f"   • Автоматическое обнаружение аномалий: {'Да' if args.auto_anomaly_detection else 'Нет'}")
    if args.auto_anomaly_detection:
        print(f"   • Порог обнаружения аномалий: {args.anomaly_threshold}")
    
    # Запускаем прогнозирование
    try:
        run_forecasting()
        
        # Дополнительный анализ результатов
        print("\n" + "=" * 80)
        print("ЭТАП 3: АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("=" * 80)
        
        analyze_results(args.output_dir)
        
    except Exception as e:
        print(f"❌ Ошибка при прогнозировании: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("ЗАВЕРШЕНИЕ РАБОТЫ")
    print("=" * 80)
    print(f"✅ Все результаты сохранены в директории: {args.output_dir}")
    print("\n📂 Созданные файлы:")
    
    # Выводим список созданных файлов
    output_files = [
        "demand_before_anomaly_fix.png",
        "anomaly_before_fix.png", 
        "anomaly_after_fix.png",
        "total_demand.png",
        "seasonal_decomposition.png",
        "monthly_seasonality.png",
        "top_industries.png",
        "industry_demand_forecast.csv",
        "random_forest_metrics.csv",
        "gradient_boosting_metrics.csv",
        "best_models_per_industry.csv",
        "model_comparison.png",
        "final_report.txt"
    ]
    
    for file_name in output_files:
        file_path = os.path.join(args.output_dir, file_name)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"   ✓ {file_name} ({file_size:,} байт)")
        else:
            print(f"   ⚠ {file_name} (не создан)")
    
    print(f"\n🎯 Основные результаты:")
    try:
        # Читаем и выводим ключевые метрики из отчета
        report_path = os.path.join(args.output_dir, 'final_report.txt')
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
                # Извлекаем ключевые строки
                lines = report_content.split('\n')
                for line in lines:
                    if 'Лучший R²:' in line or 'Лучшая модель по среднему RMSE:' in line or 'Лучшая конфигурация:' in line:
                        print(f"   {line.strip()}")
    except Exception as e:
        print(f"   Не удалось прочитать отчет: {e}")

def analyze_results(output_dir):
    """Дополнительный анализ результатов"""
    print("📈 Проведение дополнительного анализа результатов...")
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Анализ метрик
        rf_metrics_path = os.path.join(output_dir, 'random_forest_metrics.csv')
        gb_metrics_path = os.path.join(output_dir, 'gradient_boosting_metrics.csv')
        
        if os.path.exists(rf_metrics_path) and os.path.exists(gb_metrics_path):
            rf_metrics = pd.read_csv(rf_metrics_path, index_col=0)
            gb_metrics = pd.read_csv(gb_metrics_path, index_col=0)
            
            print("📊 Сводка метрик качества:")
            print(f"   Random Forest - средний R²: {rf_metrics['R2'].mean():.4f}")
            print(f"   XGBoost - средний R²: {gb_metrics['R2'].mean():.4f}")
            print(f"   Random Forest - средний RMSE: {rf_metrics['RMSE'].mean():.2f}")
            print(f"   XGBoost - средний RMSE: {gb_metrics['RMSE'].mean():.2f}")
            
            # Создание детального сравнения
            plt.figure(figsize=(15, 5))
            
            # Subplot 1: R² по отраслям
            plt.subplot(1, 3, 1)
            industries = rf_metrics.index
            x_pos = range(len(industries))
            
            plt.bar([x - 0.2 for x in x_pos], rf_metrics['R2'], 0.4, label='Random Forest', alpha=0.7)
            plt.bar([x + 0.2 for x in x_pos], gb_metrics['R2'], 0.4, label='XGBoost', alpha=0.7)
            
            plt.xlabel('Отрасли')
            plt.ylabel('R²')
            plt.title('R² по отраслям')
            plt.xticks(x_pos, industries, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: RMSE по отраслям
            plt.subplot(1, 3, 2)
            plt.bar([x - 0.2 for x in x_pos], rf_metrics['RMSE'], 0.4, label='Random Forest', alpha=0.7)
            plt.bar([x + 0.2 for x in x_pos], gb_metrics['RMSE'], 0.4, label='XGBoost', alpha=0.7)
            
            plt.xlabel('Отрасли')
            plt.ylabel('RMSE')
            plt.title('RMSE по отраслям')
            plt.xticks(x_pos, industries, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 3: MAE по отраслям
            plt.subplot(1, 3, 3)
            plt.bar([x - 0.2 for x in x_pos], rf_metrics['MAE'], 0.4, label='Random Forest', alpha=0.7)
            plt.bar([x + 0.2 for x in x_pos], gb_metrics['MAE'], 0.4, label='XGBoost', alpha=0.7)
            
            plt.xlabel('Отрасли')
            plt.ylabel('MAE')
            plt.title('MAE по отраслям')
            plt.xticks(x_pos, industries, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'detailed_metrics_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("   ✓ Создан детальный график сравнения метрик")
        
        # Анализ прогнозов
        forecast_path = os.path.join(output_dir, 'industry_demand_forecast.csv')
        if os.path.exists(forecast_path):
            forecast_df = pd.read_csv(forecast_path, index_col=0)
            forecast_df.index = pd.to_datetime(forecast_df.index)
            
            print(f"📅 Прогнозы созданы для {len(forecast_df.columns)} показателей")
            print(f"   Горизонт прогноза: {forecast_df.index[0].strftime('%Y-%m')} - {forecast_df.index[-1].strftime('%Y-%m')}")
            
            if 'total_demand' in forecast_df.columns:
                total_forecast = forecast_df['total_demand']
                print(f"   Прогноз общего спроса: {total_forecast.iloc[0]:,.0f} - {total_forecast.iloc[-1]:,.0f} рабочих мест")
                print(f"   Средний месячный спрос: {total_forecast.mean():,.0f} рабочих мест")
            
            # Создание сводного графика прогнозов
            plt.figure(figsize=(15, 8))
            
            # Строим прогнозы для всех отраслей
            for col in forecast_df.columns:
                if col != 'total_demand':
                    plt.plot(forecast_df.index, forecast_df[col], label=col, alpha=0.7)
                else:
                    plt.plot(forecast_df.index, forecast_df[col], label=col, linewidth=3, color='black')
            
            plt.title('Прогнозы потребности в инженерных кадрах по отраслям')
            plt.xlabel('Период')
            plt.ylabel('Количество рабочих мест')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'all_forecasts_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("   ✓ Создан сводный график всех прогнозов")
        
    except Exception as e:
        print(f"⚠️ Ошибка при дополнительном анализе: {e}")

if __name__ == "__main__":
    run_pipeline()
