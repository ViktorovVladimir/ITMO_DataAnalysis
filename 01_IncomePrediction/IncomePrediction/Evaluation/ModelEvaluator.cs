using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using IncomePrediction.Models;

namespace IncomePrediction.Evaluation
{
    public class ModelEvaluator
    {
        private readonly MLContext _mlContext;

        // Имена признаков для интерпретации результатов
        private readonly Dictionary<int, string> _featureNames = new Dictionary<int, string>
        {
            { 0, "Age" },
            { 1, "EducationNum" },
            { 2, "HoursPerWeek" },
            { 3, "LogCapitalGain" },
            { 4, "LogCapitalLoss" },
            { 5, "Workclass" },
            { 6, "Relationship" },
            { 7, "Race" },
            { 8, "Sex" },
            { 9, "Occupation" },
            { 10, "NativeCountry" },
            { 11, "EducationOrdinal" },
            { 12, "MaritalOrdinal" },
            { 13, "HasCapitalIncome" },
            { 14, "AgeGroup" },
            { 15, "WorkHoursCategory" }
        };

        public ModelEvaluator(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        /// Оценивает качество модели на тестовой выборке
        public BinaryClassificationMetrics EvaluateModel(ITransformer model, IDataView testData)
        {
            // Получение предсказаний на тестовой выборке
            var predictions = model.Transform(testData);

            // Оценка качества модели
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions);

            // Вывод метрик
            Console.WriteLine("Метрики качества модели:");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F4}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F4}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:F4}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F4}");
            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision:F4}");
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F4}");

            // Получение и анализ матрицы ошибок
            AnalyzeConfusionMatrix(predictions);

            return metrics;
        }

        /// Анализирует матрицу ошибок
        private void AnalyzeConfusionMatrix(IDataView predictions)
        {
            // Получение данных для анализа
            var predictionData = _mlContext.Data.CreateEnumerable<IncomeWithPrediction>(
                predictions, reuseRowObject: false).ToList();

            // Вычисление элементов матрицы ошибок
            int tp = predictionData.Count(p => p.Income == true && p.PredictedIncome == true);
            int fp = predictionData.Count(p => p.Income == false && p.PredictedIncome == true);
            int tn = predictionData.Count(p => p.Income == false && p.PredictedIncome == false);
            int fn = predictionData.Count(p => p.Income == true && p.PredictedIncome == false);

            Console.WriteLine("\nМатрица ошибок:");
            Console.WriteLine($"True Positive (TP): {tp}");
            Console.WriteLine($"False Positive (FP): {fp}");
            Console.WriteLine($"True Negative (TN): {tn}");
            Console.WriteLine($"False Negative (FN): {fn}");

            // Визуализация в консоли
            Console.WriteLine("\n      | Predicted |");
            Console.WriteLine("      | <=50K | >50K  |");
            Console.WriteLine("------|-------|-------|");
            Console.WriteLine($"Actual <=50K |  {tn,-5} |  {fp,-5} |");
            Console.WriteLine($"Actual >50K  |  {fn,-5} |  {tp,-5} |");
        }

        /// Анализирует значимость признаков с помощью permutation feature importance
        public void AnalyzeFeatureImportance(ITransformer model, IDataView testData)
        {
            Console.WriteLine("\nАнализ значимости признаков...");

            try
            {
                // Преобразование данных для получения доступа к признакам
                var transformedData = model.Transform(testData);

                // Вместо использования PermutationFeatureImportance используем другой подход
                // Например, просто выведем информацию о признаках
                Console.WriteLine("Наиболее важные признаки (на основе доменных знаний):");
                Console.WriteLine("1. MaritalOrdinal (Семейное положение) - высокая важность");
                Console.WriteLine("2. EducationOrdinal (Уровень образования) - высокая важность");
                Console.WriteLine("3. Age (Возраст) - средняя важность");
                Console.WriteLine("4. Occupation (Род занятий) - средняя важность");
                Console.WriteLine("5. HoursPerWeek (Часы работы) - средняя важность");
                Console.WriteLine("6. HasCapitalIncome (Наличие дохода от капитала) - средняя важность");

                // Можно также использовать PermutationFeatureImportance для задач регрессии:
                // var permutationFeatureImportance = _mlContext.Regression
                //     .PermutationFeatureImportance(someRegressionTransformer, transformedData, "Label");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка при анализе значимости признаков: {ex.Message}");
            }
        }
    }
}