using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace mlnet1
{
    public class Mushroom
    {
        public string @class { get; set; }
        public string cap_shape { get; set; }
        public string cap_surface { get; set; }
        public string cap_color { get; set; }
        public string bruises { get; set; }
        public string odor { get; set; }
        public string gill_attachment { get; set; }
        public string gill_spacing { get; set; }
        public string gill_size { get; set; }
        public string gill_color { get; set; }
        public string stalk_shape { get; set; }
        public string stalk_root { get; set; }
        public string stalk_surface_above_ring { get; set; }
        public string stalk_surface_below_ring { get; set; }
        public string stalk_color_above_ring { get; set; }
        public string stalk_color_below_ring { get; set; }
        public string veil_type { get; set; }
        public string veil_color { get; set; }
        public string ring_number { get; set; }
        public string ring_type { get; set; }
        public string spore_print_color { get; set; }
        public string population { get; set; }
        public string habitat { get; set; }
    }

    public class MushroomPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsEdible;

        [ColumnName("Score")]
        public float Score;
    }


    internal class Program
    {
        private static void Main(string[] args)
        {
            var dataPath = "mushrooms.csv";
            var mlContext = new MLContext();
            var columns = new[]
                {
                    new TextLoader.Column("class", DataKind.Text, 0),
                    new TextLoader.Column("cap_shape", DataKind.Text, 1),
                    new TextLoader.Column("cap_surface", DataKind.Text, 2),
                    new TextLoader.Column("cap_color", DataKind.Text, 3),
                    new TextLoader.Column("bruises", DataKind.Text, 4),
                    new TextLoader.Column("odor", DataKind.Text, 5),
                    new TextLoader.Column("gill_attachment", DataKind.Text, 6),
                    new TextLoader.Column("gill_spacing", DataKind.Text, 7),
                    new TextLoader.Column("gill_size", DataKind.Text, 8),
                    new TextLoader.Column("gill_color", DataKind.Text, 9),
                    new TextLoader.Column("stalk_shape", DataKind.Text, 10),
                    new TextLoader.Column("stalk_root", DataKind.Text, 11),
                    new TextLoader.Column("stalk_surface_above_ring", DataKind.Text, 12),
                    new TextLoader.Column("stalk_surface_below_ring", DataKind.Text, 13),
                    new TextLoader.Column("stalk_color_above_ring", DataKind.Text, 14),
                    new TextLoader.Column("stalk_color_below_ring", DataKind.Text, 15),
                    new TextLoader.Column("veil_type", DataKind.Text, 16),
                    new TextLoader.Column("veil_color", DataKind.Text, 17),
                    new TextLoader.Column("ring_number", DataKind.Text, 18),
                    new TextLoader.Column("ring_type", DataKind.Text, 19),
                    new TextLoader.Column("spore_print_color", DataKind.Text, 20),
                    new TextLoader.Column("population", DataKind.Text, 21),
                    new TextLoader.Column("habitat", DataKind.Text, 22),
                };

            TextLoader reader = mlContext.Data.CreateTextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = columns
            });

            var fullData = reader.Read(dataPath);
            (var trainingDataView, var testDataView) = mlContext.BinaryClassification.TrainTestSplit(fullData, 0.3, seed: 5555);

            var pipeline =
               mlContext.Transforms.Conversion.ValueMap(new[] { "e".AsMemory(), "p".AsMemory() }, new[] { true, false }, ("class", "class"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("cap_shape","cap_shape_encoded"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("cap_surface"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("cap_color"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("bruises"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("odor"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("gill_attachment"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("gill_spacing"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("gill_size"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("gill_color"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("stalk_shape"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("stalk_root"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("stalk_surface_above_ring"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("stalk_surface_below_ring"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("stalk_color_above_ring"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("stalk_color_below_ring"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("veil_type"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("veil_color"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("ring_number"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("ring_type"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("spore_print_color"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("population"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("habitat"))
                                    .Append(mlContext.Transforms.Concatenate("Features", "cap_shape_encoded",
                                                                                            "cap_surface",
                                                                                            "cap_color",
                                                                                            "bruises",
                                                                                            "odor",
                                                                                            "gill_attachment",
                                                                                            "gill_spacing",
                                                                                            "gill_size",
                                                                                            "gill_color",
                                                                                            "stalk_shape",
                                                                                            "stalk_root",
                                                                                            "stalk_surface_above_ring",
                                                                                            "stalk_surface_below_ring",
                                                                                            "stalk_color_above_ring",
                                                                                            "stalk_color_below_ring",
                                                                                            "veil_type",
                                                                                            "veil_color",
                                                                                            "ring_number",
                                                                                            "ring_type",
                                                                                            "spore_print_color",
                                                                                            "population",
                                                                                            "habitat"));

           
            var trainer = mlContext.BinaryClassification.Trainers.FastTree(labelColumn: "class", featureColumn: "Features");
            var trainingPipeline = pipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            using (var fs = new FileStream("model.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(trainedModel, fs);
            }


            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "class", "Score");

            Console.WriteLine($"*Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"*F1Score:  {metrics.F1Score:P2}");

            // ---------------- TEKRAR KULLANIM
            var context2 = new MLContext();
            ITransformer loadedModel;
            using (var stream = File.OpenRead("model.zip"))
            {
                loadedModel = context2.Model.Load(stream);
            }

            var classifier = loadedModel.CreatePredictionEngine<Mushroom, MushroomPrediction>(mlContext);
            var result = classifier.Predict(new Mushroom
            {
                cap_shape = "x",
                cap_surface = "s",
                cap_color = "y",
                bruises = "t",
                odor = "a",
                gill_attachment = "f",
                gill_spacing = "c",
                gill_size = "b",
                gill_color = "k",
                stalk_shape = "e",
                stalk_root = "c",
                stalk_surface_above_ring = "s",
                stalk_surface_below_ring = "s",
                stalk_color_above_ring = "w",
                stalk_color_below_ring = "w",
                veil_type = "p",
                veil_color = "w",
                ring_number = "o",
                ring_type = "p",
                spore_print_color = "n",
                population = "n",
                habitat = "g",
            });

            Console.WriteLine(result.IsEdible);
            Console.WriteLine(result.Score);
            Console.ReadLine();
        }
    }
}
