using Microsoft.ML.Data;

namespace IncomePrediction.Models
{
    /// Класс для представления результатов предсказания
    public class AdultPrediction
    {
        // Предсказанное значение (доход >50K или <=50K)
        [ColumnName("PredictedLabel")]
        public bool PredictedIncome { get; set; }
        
        // Вероятность того, что доход >50K
        public float Probability { get; set; }
        
        // Значение оценки перед преобразованием в вероятность
        public float Score { get; set; }
    }
    
    /// Класс для совместного хранения фактических и предсказанных значений
    public class IncomeWithPrediction
    {
        // Фактическое значение дохода
        public bool Income { get; set; }
        
        // Предсказанное значение дохода
        public bool PredictedIncome { get; set; }
        
        // Вероятность предсказания
        public float Probability { get; set; }
    }
}