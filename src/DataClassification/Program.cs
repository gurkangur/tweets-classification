using Microsoft.ML;
using System;
using System.IO;

namespace DataClassification
{
    class Program
    {
        // <SnippetDeclareGlobalVariables>
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "tweets_train.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "tweets_test.tsv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<Tweet, TweetPrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;
        // </SnippetDeclareGlobalVariables>
        static void Main(string[] args)
        {
            // Create MLContext to be shared across the model creation workflow objects
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            // <SnippetCreateMLContext>
            _mlContext = new MLContext(seed: 0);
            // </SnippetCreateMLContext>

            Console.WriteLine($"=============== Loading Dataset  ===============");

            // <SnippetLoadTrainData>
            _trainingDataView = _mlContext.Data.LoadFromTextFile<Tweet>(_trainDataPath, hasHeader: true);
            // </SnippetLoadTrainData>

            Console.WriteLine($"=============== Finished Loading Dataset  ===============");

            // <SnippetSplitData>
            //   var (trainData, testData) = _mlContext.MulticlassClassification.TrainTestSplit(_trainingDataView, testFraction: 0.1);
            // </SnippetSplitData>

            // <SnippetCallProcessData>
            var pipeline = ProcessData();
            // </SnippetCallProcessData>

            // <SnippetCallBuildAndTrainModel>
            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
            // </SnippetCallBuildAndTrainModel>

            // <SnippetCallEvaluate>
            Evaluate(_trainingDataView.Schema);
            // </SnippetCallEvaluate>

            // <SnippetCallPredictTweet>
            PredictTweet();
            // </SnippetCallPredictTweet>
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            Console.WriteLine($"=============== Processing Data ===============");
            // STEP 2: Common data process configuration with pipeline data transformations
            // <SnippetMapValueToKey>
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Category", outputColumnName: "Label")
                            // </SnippetMapValueToKey>
                            // <SnippetFeaturizeText>
                            .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "FullText", outputColumnName: "FullTextFeaturized"))
                            // </SnippetFeaturizeText>
                            // <SnippetConcatenate>
                            .Append(_mlContext.Transforms.Concatenate("Features", "FullTextFeaturized"))
                            // </SnippetConcatenate>
                            //Sample Caching the DataView so estimators iterating over the data multiple times, instead of always reading from file, using the cache might get better performance.
                            // <SnippetAppendCache>
                            .AppendCacheCheckpoint(_mlContext);
            // </SnippetAppendCache>

            Console.WriteLine($"=============== Finished Processing Data ===============");

            // <SnippetReturnPipeline>
            return pipeline;
            // </SnippetReturnPipeline>
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            // STEP 3: Create the training algorithm/trainer
            // Use the multi-class SDCA algorithm to predict the label using features.
            //Set the trainer/algorithm and map label to value (original readable state)
            // <SnippetAddTrainer>
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            // </SnippetAddTrainer>

            // STEP 4: Train the model fitting to the DataSet
            Console.WriteLine($"=============== Training the model  ===============");

            // <SnippetTrainModel>
            _trainedModel = trainingPipeline.Fit(trainingDataView);
            // </SnippetTrainModel>
            Console.WriteLine($"=============== Finished Training the model Ending time: {DateTime.Now.ToString()} ===============");

            // (OPTIONAL) Try/test a single prediction with the "just-trained model" (Before saving the model)
            Console.WriteLine($"=============== Single Prediction just-trained-model ===============");

            // Create prediction engine related to the loaded trained model
            // <SnippetCreatePredictionEngine1>
            _predEngine = _mlContext.Model.CreatePredictionEngine<Tweet, TweetPrediction>(_trainedModel);
            // </SnippetCreatePredictionEngine1>
            // <SnippetCreateTestTweet1>
            Tweet Tweet = new Tweet()
            {
                FullText = "Is there any iPhone emulator which I can use to test how websites look in iPhone browser?"
            };
            // </SnippetCreateTestTweet1>

            // <SnippetPredict>
            var prediction = _predEngine.Predict(Tweet);
            // </SnippetPredict>

            // <SnippetOutputPrediction>
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Category} ===============");
            // </SnippetOutputPrediction>

            // <SnippetReturnModel>
            return trainingPipeline;
            // </SnippetReturnModel>
        }

        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            // STEP 5:  Evaluate the model in order to get the model's accuracy metrics
            Console.WriteLine($"=============== Evaluating to get model's accuracy metrics - Starting time: {DateTime.Now.ToString()} ===============");

            //Load the test dataset into the IDataView
            // <SnippetLoadTestDataset>
            var testDataView = _mlContext.Data.LoadFromTextFile<Tweet>(_testDataPath, hasHeader: true);
            // </SnippetLoadTestDataset>

            //Evaluate the model on a test dataset and calculate metrics of the model on the test data.
            // <SnippetEvaluate>
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));
            // </SnippetEvaluate>

            Console.WriteLine($"=============== Evaluating to get model's accuracy metrics - Ending time: {DateTime.Now.ToString()} ===============");
            // <SnippetDisplayMetrics>
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
            // </SnippetDisplayMetrics>

            // Save the new model to .ZIP file
            // <SnippetCallSaveModel>
            SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
            // </SnippetCallSaveModel>
        }

        public static void PredictTweet()
        {
            // <SnippetLoadModel>
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
            // </SnippetLoadModel>

            // <SnippetAddTestTweet>
            Tweet singleTweet = new Tweet() { FullText = "Free online course on developing android applications" };
            // </SnippetAddTestTweet>

            //Predict label for single hard-coded Tweet
            // <SnippetCreatePredictionEngine>
            _predEngine = _mlContext.Model.CreatePredictionEngine<Tweet, TweetPrediction>(loadedModel);
            // </SnippetCreatePredictionEngine>

            // <SnippetPredictTweet>
            var prediction = _predEngine.Predict(singleTweet);
            // </SnippetPredictTweet>

            // <SnippetDisplayResults>
            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Category} ===============");
            // </SnippetDisplayResults>
        }

        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            // <SnippetSaveModel>
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
            // </SnippetSaveModel>

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
    }
}
