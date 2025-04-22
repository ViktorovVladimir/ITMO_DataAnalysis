using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using IncomePrediction.Models;

namespace IncomePrediction.Training
{
    public class ModelTrainer
    {
        private readonly MLContext _mlContext;
        
        public ModelTrainer(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        /// <summary>
        /// -- Trains a simplified logistic regression model
        /// </summary>
        public ITransformer TrainAndCompareModels(IEstimator<ITransformer> dataPrepPipeline, IDataView trainData)
        {
            Console.WriteLine("\t*Training a simplified model:");

            //--. We use the simplest call without additional parameters
            var estimator = _mlContext.BinaryClassification.Trainers.SdcaLogisticRegression();
            
            Console.WriteLine("\t*Building a pipeline...");

            //--. Creating a complete pipeline (preprocessing + training)
            var pipeline = dataPrepPipeline.Append( estimator );

            //--. Training a model with time dimension
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            
            Console.WriteLine("\t*Start of Fit() operation...");
            var model = pipeline.Fit(trainData);
            stopWatch.Stop();
            
            Console.WriteLine("\n\tThe Fit() operation has completed.");
            Console.WriteLine($"Training time: {stopWatch.ElapsedMilliseconds / 1000.0:F2} seconds");

            //--. Performing a simple assessment
            Console.WriteLine("Performing Model Evaluation...");
            var predictions = model.Transform( trainData );
            var metrics = _mlContext.BinaryClassification.Evaluate( predictions );
            
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F4}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F4}");
            
            return model;
        }
    }
}