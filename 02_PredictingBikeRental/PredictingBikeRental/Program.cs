using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using PredictingBikeRental.DataProcessing;
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
                // TODO: Реализовать обучение нескольких моделей и сравнить их
                //--. Step 3: Training a simplified model
                // 6. Оценка качества модели
                // 7. Выполнение предсказаний
                Console.WriteLine("\nStep 3: Model training...");
                Console.WriteLine("------");
                var modelTrainer = new ModelTrainer( mlContext );
                //var model = modelTrainer.TrainAndCompareModels( dataPrepPipeline, trainTestData.TrainSet );
                var bestModel = modelTrainer.TrainMultipleModels(dataPrepPipeline, trainTestData.TrainSet);

                //IDataView newData = ... 
                //MakePredictions(bestModel, newData);


                Console.WriteLine("Training completed successfully!");
            }
            catch( Exception ex )
            {
                Console.WriteLine($"\nError: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }

            Console.WriteLine("\nPress any key to complete...");
            Console.ReadKey();
        }
    }
}

