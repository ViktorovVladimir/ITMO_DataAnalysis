using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using PredictingBikeRental.DataProcessing;
using PredictingBikeRental.Models;
using PredictingBikeRental.Training;

namespace PredictingBikeRental
{
    class Program
    {
        //--. Path to data files
        private static string _dataPath = "../../Data/bike_sharing.csv";

        //--.
        static void Main(string[] args)
        {
            Console.WriteLine("====================================================");
            Console.WriteLine("    Predicting Bike Rental Type Using ML.NET        ");
            Console.WriteLine("====================================================\n");

            // --. Creating ML.NET context
            var mlContext = new MLContext( seed: 0 );
            
            //--.
            try
            {
                //--. Step 1: Data loading and analysis
                Console.WriteLine("Step 1: Data loading and analysis...");
                Console.WriteLine("------");
                var dataProcessor = new DataProcessor( mlContext );
                var data = dataProcessor.LoadData( _dataPath );

                //--. Let's limit the data for quick debugging
                var sampleData = mlContext.Data.TakeRows( data, 1000 );
                dataProcessor.ExploreData( sampleData );

                
                //--. Step 2: Data separation and pipeline creation
                Console.WriteLine("\nStep 2: Data separation and processing pipeline creation...");
                Console.WriteLine("------");
                var trainTestData = dataProcessor.SplitData( sampleData );
                // --. Creating a data processing pipeline
                var dataPrepPipeline = dataProcessor.CreateDataProcessingPipeline();

                // 5. Обучение моделей и выбор лучшей
                //--. Step 3: Training a simplified model
                // 6. Оценка качества модели
                Console.WriteLine("\nStep 3: Model training...");
                Console.WriteLine("        *Training models and choosing the best one...");
                Console.WriteLine("------");
                var modelTrainer = new ModelTrainer( mlContext );
                //var model = modelTrainer.TrainAndCompareModels( dataPrepPipeline, trainTestData.TrainSet );
                var bestModel = modelTrainer.TrainMultipleModels( dataPrepPipeline, trainTestData.TrainSet, trainTestData.TestSet );
                
                Console.WriteLine("\nTraining completed successfully!\n");



                // 7. Выполнение предсказаний
                //--. Step 4: Making predictions
                //4.1. Preparing a model for prediction
                Console.WriteLine("\nStep 4: Making predictions...");
                Console.WriteLine("------");


                var predictor = mlContext.Model.CreatePredictionEngine<BikeRentalData, RentalTypePrediction>(bestModel);
                
                //4.2. We prepare any of our data for which we will receive a prediction
                var sample = new BikeRentalData
                {
                    Season = 3,
                    Month = 7,
                    Hour = 18,
                    Holiday = 0,
                    Weekday = 2,
                    WorkingDay = 1,
                    WeatherCondition = 1,
                    Temperature = 26,
                    Humidity = 55,
                    Windspeed = 10
                };

                Console.WriteLine("\t*My data: " + sample.ToString() + "\n");


                //4.3. We send our data to the model
                var prediction = predictor.Predict( sample );

                //4.4. Displaying the prediction
                Console.WriteLine($"\t*Rental type: {(prediction.PredictedRentalType ? "long-term" : "short-term")}, " +
                                    $"probability = {prediction.Probability:P1}");



                //--. Step 5: Save the model for later use without wasting time on training
                Console.WriteLine("\nStep 5: Save the model for later use without wasting time on training...");
                Console.WriteLine("------");
                mlContext.Model.Save( bestModel, trainTestData.TrainSet.Schema, "BikeRentalModel.zip" );
                Console.WriteLine("\t*Saving the model completed successfully!\n");



            }
            catch ( Exception ex )
            {
                Console.WriteLine($"\nError: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }

            Console.WriteLine("\nPress any key to complete...");
            Console.ReadKey();
        }
    }
}

