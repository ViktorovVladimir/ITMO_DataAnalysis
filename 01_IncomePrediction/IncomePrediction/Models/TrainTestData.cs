using Microsoft.ML;
using Microsoft.ML.Data;

namespace IncomePrediction.Models
{
    /// Класс для хранения разделенных обучающей и тестовой выборок
    public class TrainTestData
    {
        public IDataView TrainSet { get; set; }
        public IDataView TestSet { get; set; }
        
        public TrainTestData(IDataView trainSet, IDataView testSet)
        {
            TrainSet = trainSet;
            TestSet = testSet;
        }
    }
}