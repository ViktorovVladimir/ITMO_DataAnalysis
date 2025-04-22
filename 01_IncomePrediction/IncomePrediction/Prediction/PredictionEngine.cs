using System;
using System.IO;
using Microsoft.ML;
using IncomePrediction.Models;

namespace IncomePrediction.Prediction
{
    public class PredictionEngine
    {
        private readonly MLContext _mlContext;
        private ITransformer? _loadedModel;  // Добавляем ? для nullable
        private PredictionEngine<AdultData, AdultPrediction>? _predictionEngine;  // Добавляем ? для nullable

        public PredictionEngine(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        /// Загружает модель из файла
        public void LoadModel(string modelPath)
        {
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Файл модели не найден: {modelPath}");
            }

            // Загрузка модели
            DataViewSchema modelSchema;
            _loadedModel = _mlContext.Model.Load(modelPath, out modelSchema);

            // Создание движка прогнозирования
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<AdultData, AdultPrediction>(_loadedModel);

            Console.WriteLine($"Модель загружена из файла: {modelPath}");
        }

        /// Демонстрирует прогнозирование на предопределенных примерах
        public void DemonstratePredictions()
        {
            Console.WriteLine("Демонстрация предсказаний на типовых примерах:");

            // Пример 1: Высокооплачиваемый специалист
            var example1 = new AdultData
            {
                Age = 45,
                Workclass = "Private",
                Education = "Bachelors",
                EducationNum = 13,
                MaritalStatus = "Married-civ-spouse",
                Occupation = "Exec-managerial",
                Relationship = "Husband",
                Race = "White",
                Sex = "Male",
                CapitalGain = 15000,
                CapitalLoss = 0,
                HoursPerWeek = 60,
                NativeCountry = "United-States"
            };

            var prediction1 = _predictionEngine.Predict(example1);
            Console.WriteLine("\nПример 1: Высокооплачиваемый специалист");
            Console.WriteLine("Характеристики:");
            Console.WriteLine("- 45 лет, мужчина, женат");
            Console.WriteLine("- Высшее образование (бакалавр)");
            Console.WriteLine("- Руководящая должность в частной компании");
            Console.WriteLine("- Прирост капитала: $15,000");
            Console.WriteLine("- 60 часов работы в неделю");
            Console.WriteLine($"Предсказание: {(prediction1.PredictedIncome ? ">50K" : "<=50K")}");
            Console.WriteLine($"Вероятность дохода >50K: {prediction1.Probability:P2}");

            // Пример 2: Низкооплачиваемый работник
            var example2 = new AdultData
            {
                Age = 25,
                Workclass = "Private",
                Education = "HS-grad",
                EducationNum = 9,
                MaritalStatus = "Never-married",
                Occupation = "Service",
                Relationship = "Not-in-family",
                Race = "Black",
                Sex = "Female",
                CapitalGain = 0,
                CapitalLoss = 0,
                HoursPerWeek = 35,
                NativeCountry = "United-States"
            };

            var prediction2 = _predictionEngine.Predict(example2);
            Console.WriteLine("\nПример 2: Низкооплачиваемый работник");
            Console.WriteLine("Характеристики:");
            Console.WriteLine("- 25 лет, женщина, не замужем");
            Console.WriteLine("- Среднее образование");
            Console.WriteLine("- Сфера обслуживания в частной компании");
            Console.WriteLine("- Нет прироста капитала");
            Console.WriteLine("- 35 часов работы в неделю");
            Console.WriteLine($"Предсказание: {(prediction2.PredictedIncome ? ">50K" : "<=50K")}");
            Console.WriteLine($"Вероятность дохода >50K: {prediction2.Probability:P2}");
        }

        /// Запускает интерактивный режим предсказаний
        public void RunInteractivePredictions()
        {
            Console.WriteLine("\n=== Интерактивное прогнозирование дохода ===");
            Console.WriteLine("Введите данные для предсказания или 'exit' для выхода\n");

            while (true)
            {
                Console.WriteLine("\nВведите информацию о человеке:");

                Console.Write("Возраст (например, 35): ");
                string ageInput = Console.ReadLine();
                if (string.Equals(ageInput, "exit", StringComparison.OrdinalIgnoreCase))
                    break;

                Console.Write("Класс работы (например, Private, Self-emp, Federal-gov): ");
                string workclass = Console.ReadLine();

                Console.Write("Образование (например, Bachelors, HS-grad, Masters): ");
                string education = Console.ReadLine();

                Console.Write("Количество лет образования (например, 13): ");
                string eduNumInput = Console.ReadLine();

                Console.Write("Семейное положение (например, Married-civ-spouse, Never-married): ");
                string maritalStatus = Console.ReadLine();

                Console.Write("Род занятий (например, Exec-managerial, Prof-specialty): ");
                string occupation = Console.ReadLine();

                Console.Write("Пол (Male/Female): ");
                string sex = Console.ReadLine();

                Console.Write("Часов работы в неделю (например, 40): ");
                string hoursInput = Console.ReadLine();

                Console.Write("Прирост капитала ($): ");
                string capitalGainInput = Console.ReadLine();

                Console.Write("Потери капитала ($): ");
                string capitalLossInput = Console.ReadLine();

                // Создание объекта с данными
                var inputData = new AdultData
                {
                    Age = float.TryParse(ageInput, out var age) ? age : 0,
                    Workclass = workclass,
                    Education = education,
                    EducationNum = float.TryParse(eduNumInput, out var eduNum) ? eduNum : 0,
                    MaritalStatus = maritalStatus,
                    Occupation = occupation,
                    Sex = sex,
                    HoursPerWeek = float.TryParse(hoursInput, out var hours) ? hours : 0,
                    CapitalGain = float.TryParse(capitalGainInput, out var gain) ? gain : 0,
                    CapitalLoss = float.TryParse(capitalLossInput, out var loss) ? loss : 0,
                    // Остальные поля заполняем значениями по умолчанию
                    Relationship = "Unknown",
                    Race = "Unknown",
                    NativeCountry = "United-States"
                };

                // Получение предсказания
                var prediction = _predictionEngine.Predict(inputData);

                // Вывод результата
                Console.WriteLine("\n--- Результат предсказания ---");
                Console.WriteLine($"Предсказанный уровень дохода: {(prediction.PredictedIncome ? ">50K" : "<=50K")}");
                Console.WriteLine($"Вероятность дохода >50K: {prediction.Probability:P2}");
                Console.WriteLine($"Уверенность модели: {(prediction.Probability > 0.5 ? prediction.Probability : 1 - prediction.Probability):P2}");
            }
        }
    }
}