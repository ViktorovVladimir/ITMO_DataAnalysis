using System;
using System.Diagnostics;
using System.IO;
using Microsoft.ML;
using IncomePrediction.Models;
using IncomePrediction.DataProcessing;
using IncomePrediction.Training;
using IncomePrediction.Evaluation;
using IncomePrediction.Prediction;

namespace IncomePrediction
{
    class Program
    {
        //--. Path to data files
        private static string _dataPath = "../../../Data/adult.csv";
        
        static void Main(string[] args)
        {
            Console.WriteLine("====================================================");
            Console.WriteLine("    Income forecasting based on demographic data    ");
            Console.WriteLine("====================================================\n");

            //--. Creating an ML.NET context with a fixed seed for reproducible results
            var mlContext = new MLContext(seed: 42);

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
                var dataPrepPipeline = dataProcessor.CreateDataProcessingPipeline();

                //--. Step 3: Training a simplified model
                Console.WriteLine("\nStep 3: Model training...");
                Console.WriteLine("------");
                var modelTrainer = new ModelTrainer( mlContext );
                var model = modelTrainer.TrainAndCompareModels( dataPrepPipeline, trainTestData.TrainSet );
                
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