using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataClassification
{

    public class Tweet
    {
        [LoadColumn(0)]
        public string Id { get; set; }
        [LoadColumn(1)]
        public string Category { get; set; }
        [LoadColumn(2)]
        public string FullText { get; set; }
    }

    public class TweetPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Category;
    }
}
