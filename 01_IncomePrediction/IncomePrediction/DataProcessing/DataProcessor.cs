using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using IncomePrediction.Models;

namespace IncomePrediction.DataProcessing
{
    public class DataProcessor
    {
        private readonly MLContext _mlContext;

        public DataProcessor(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        /// <summary>
        /// --. Loads data from a CSV file
        /// </summary>
        public IDataView LoadData( string dataPath )
        {
            //--. Loading data taking into account format features
            var data = _mlContext.Data.LoadFromTextFile<AdultData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                trimWhitespace: true);

            Console.WriteLine($"\t*Data loaded from file: {dataPath}");
            return data;
        }

        /// <summary>
        /// --. Performs exploratory data analysis
        /// </summary>
        public void ExploreData( IDataView data )
        {
            //--. Converting IDataView to a Collection for Analysis
            var adultList = _mlContext.Data
                .CreateEnumerable<AdultData>(data, reuseRowObject: false)
                .ToList();

            Console.WriteLine($"\t*Number of entries: {adultList.Count}");

            //--. Analysis of numerical features
            Console.WriteLine("\n\t*Analysis of numerical features:");
            Console.WriteLine($"\t Age: Min={adultList.Min(x => x.Age)}, Max={adultList.Max(x => x.Age)}, Avg={adultList.Average(x => x.Age):F2}");
            Console.WriteLine($"\t EducationNum: Min={adultList.Min(x => x.EducationNum)}, Max={adultList.Max(x => x.EducationNum)}, Avg={adultList.Average(x => x.EducationNum):F2}");
            Console.WriteLine($"\t HoursPerWeek: Min={adultList.Min(x => x.HoursPerWeek)}, Max={adultList.Max(x => x.HoursPerWeek)}, Avg={adultList.Average(x => x.HoursPerWeek):F2}");



            // --. Analysis of the distribution of the target variable
            var incomeGroups = adultList.GroupBy(x => x.Income).Select(g => new { Income = g.Key, Count = g.Count() });
            Console.WriteLine("\n\t*Distribution of the target variable Income:");            
            foreach( var group in incomeGroups )
            {
                Console.WriteLine($"\t  {group.Income}: {group.Count} ({(float)group.Count / adultList.Count * 100:F2}%)");
            }
        }

        /// <summary>
        /// --. Splits data into training and testing sets
        /// </summary>
        public Models.TrainTestData SplitData( IDataView data )
        {
            var mlSplitData = _mlContext.Data.TrainTestSplit( data, testFraction: 0.2 );
            var trainTestData = new Models.TrainTestData( mlSplitData.TrainSet, mlSplitData.TestSet );
            Console.WriteLine("\t*The data is divided into training and testing samples (80% / 20%)");
            return trainTestData;
        }

        /// <summary>
        /// --. Creates a simplified data processing pipeline
        /// </summary>
        public IEstimator<ITransformer> CreateDataProcessingPipeline()
        {
            Console.WriteLine("\t*Creating a simplified data processing pipeline...");

            //--. Converting Income strings to booleans directly via CustomMapping
            var dataPrepPipeline = _mlContext.Transforms.CustomMapping<AdultData, AdultDataWithBoolLabel>(
                    (input, output) =>
                    {
                        // >50K -> true, <=50K -> false
                        output.Label = input.Income.Trim() == ">50K";
                    },
                    "IncomeMapping")

                // Minimum required data transformations - for each feature separately
                .Append(_mlContext.Transforms.NormalizeMinMax("Age"))
                .Append(_mlContext.Transforms.NormalizeMinMax("EducationNum"))
                .Append(_mlContext.Transforms.NormalizeMinMax("HoursPerWeek"))

                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("WorkclassEncoded", "Workclass"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("EducationEncoded", "Education"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("MaritalStatusEncoded", "MaritalStatus"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("OccupationEncoded", "Occupation"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("RelationshipEncoded", "Relationship"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("RaceEncoded", "Race"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("SexEncoded", "Sex"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("NativeCountryEncoded", "NativeCountry"))

                // Combining all features
                .Append(_mlContext.Transforms.Concatenate("Features",
                    "Age", "EducationNum", "HoursPerWeek", "CapitalGain", "CapitalLoss",
                    "WorkclassEncoded", "EducationEncoded", "MaritalStatusEncoded", "OccupationEncoded",
                    "RelationshipEncoded", "RaceEncoded", "SexEncoded", "NativeCountryEncoded"));

            Console.WriteLine("\n\tA simplified data processing pipeline has been created");
            
            return dataPrepPipeline;
        }
    }
}