using System.Collections;
using System.Collections.Generic;
using System;
using System.IO;
using System.Linq;

namespace ML
{
    public static class UI
    {
        /*public void Train(double[,] input, double[,] yTrue, int epochs, double learningRate, int batchSize,
    Func<double, double[,], double[,], double[,], double[,], double[,], double[,], double[,]> OptimizationAlgorithm)*/
        public static void HandleSupervisedTrain(SupervisedNetwork network, double[,] input, double[,] yTrue, double[,] featureTestData, double[,] targetTestData)
        {
            int epochs = 0;
            double learningRate = 0;
            int batchSize = 0;

            Console.WriteLine();
            Console.WriteLine("Enter number of epochs: ");
            {
                string userInput = "";
                bool goodInput = false;
                while (!goodInput)
                {
                    userInput = Console.ReadLine();
                    if (int.TryParse(userInput, out int number))
                    {
                        epochs = int.Parse(userInput);
                        break;
                    }
                    else
                    {

                    }
                }
            }
            Console.WriteLine("Enter learning rate: ");
            {
                string userInput = "";
                bool goodInput = false;
                while (!goodInput)
                {
                    userInput = Console.ReadLine();
                    if (double.TryParse(userInput, out double number))
                    {
                        learningRate = double.Parse(userInput);
                        break;
                    }
                    else
                    {

                    }
                }
            }
            Console.WriteLine("Enter batch size: ");
            {
                string userInput = "";
                bool goodInput = false;
                while (!goodInput)
                {
                    userInput = Console.ReadLine();
                    if (int.TryParse(userInput, out int number))
                    {
                        batchSize = int.Parse(userInput);
                        break;
                    }
                    else
                    {

                    }
                }
            }
            Console.WriteLine("'Gradient descent' is being utilized as optimization algorithm.");
            OptimizationAlgorithm optimization = new OptimizationAlgorithm();

            network.Train(input, yTrue, epochs, learningRate, batchSize, optimization, featureTestData, targetTestData);
        }

        public static void HandleData(string filePath, string pathToCSVDirectory, ref double[,] featureTrainData,
            ref double[,] featureTestData, ref double[,] targetTrainData, ref double[,] targetTestData)
        {
            // feature_train.csv, feature_test.csv, target_train.csv, target_test.csv
            string featureTrainCSVName = "feature_train.csv";
            string featureTrainCSVPath = Path.Combine(pathToCSVDirectory, featureTrainCSVName);
            List<List<string>> featureTrainList = new List<List<string>>();

            string featureTestCSVName = "feature_test.csv";
            string featureTestCSVPath = Path.Combine(pathToCSVDirectory, featureTestCSVName);
            List<List<string>> featureTestList = new List<List<string>>();


            string targetTrainCSVName = "target_train.csv";
            string targetTrainCSVPath = Path.Combine(pathToCSVDirectory, targetTrainCSVName);
            List<List<string>> targetTrainList = new List<List<string>>();

            string targetTestCSVName = "target_test.csv";
            string targetTestCSVPath = Path.Combine(pathToCSVDirectory, targetTestCSVName);
            List<List<string>> targetTestList = new List<List<string>>();

            if (File.Exists(featureTrainCSVPath) && File.Exists(featureTestCSVPath) &&
                File.Exists(targetTrainCSVPath) && File.Exists(targetTestCSVPath))
            {
                Console.WriteLine("Data already organized and stored, would you like to override: ('Y' or 'N'): ");
                {
                    string userInput = "";

                    while (userInput != "Y" && userInput != "N")
                    {
                        userInput = Console.ReadLine();
                        if (userInput == "Y")
                        {
                            break;
                        }
                        else if (userInput == "N")
                        {
                            // fill the lists with the csv files data
                            DataFunctions.LoadCSV(featureTrainList, featureTrainCSVPath);
                            DataFunctions.LoadCSV(featureTestList, featureTestCSVPath);
                            DataFunctions.LoadCSV(targetTrainList, targetTrainCSVPath);
                            DataFunctions.LoadCSV(targetTestList, targetTestCSVPath);

                            // convert the lists to double[,]
                            featureTrainData = DataFunctions.StringMatrixToDoubleMatrix(featureTrainList);
                            featureTestData = DataFunctions.StringMatrixToDoubleMatrix(featureTestList);
                            targetTrainData = DataFunctions.StringMatrixToDoubleMatrix(targetTrainList);
                            targetTestData = DataFunctions.StringMatrixToDoubleMatrix(targetTestList);

                            Console.WriteLine("Data has been stored within arrays: Ready for training.");

                            return;
                        }
                        else
                        {

                        }
                    }
                }
            }

            List<List<string>> featureData = new List<List<string>>();
            List<List<string>> targetData = new List<List<string>>();
            List<List<string>> featureLabels = new List<List<string>>();
            List<List<string>> targetLabels = new List<List<string>>();

            List<string> featureLabel = new List<string>();
            featureLabels.Add(featureLabel);
            List<string> targetLabel = new List<string>();
            targetLabels.Add(targetLabel);


            DataFunctions.CSVToStringList(featureData, featureLabels, targetData, targetLabels, filePath);


            Console.WriteLine("-----------");
            Console.WriteLine("Raw Dataset");
            Console.WriteLine("-----------");
            Console.WriteLine();
            Console.WriteLine("Features: " + (featureData[0].Count));
            Console.WriteLine("Targets: " + targetData[0].Count);
            Console.WriteLine("Examples: " + (featureData.Count));
            Console.WriteLine();
            DataFunctions.PrintData(featureLabels, targetLabels);
            DataFunctions.PrintData(featureData, targetData);

            // Drop feature names

            Console.WriteLine();
            Console.WriteLine("Enter feature names to drop from dataset (type name and press enter or type 'DONE' and enter to stop:");
            List<string> featuresToDrop = new List<string>();

            {
                string userInput = "";

                while (userInput != "DONE")
                {
                    userInput = Console.ReadLine();
                    if (featuresToDrop.Contains(userInput))
                    {
                        continue;
                    }
                    else if (userInput == "DONE")
                    {
                        continue;
                    }
                    else if (!featureLabels[0].Contains(userInput))
                    {
                        continue;
                    }
                    else
                    {
                        featuresToDrop.Add(userInput);
                    }
                }
            }

            DataFunctions.HandleDropFeatures(featureData, featureLabels, featuresToDrop);

            Console.WriteLine();
            Console.WriteLine("------------------");
            Console.WriteLine("Dataset After Drop");
            Console.WriteLine("------------------");
            Console.WriteLine();
            Console.WriteLine("Features: " + (featureData[0].Count));
            Console.WriteLine("Targets: " + targetData[0].Count);
            Console.WriteLine("Examples: " + (featureData.Count));
            Console.WriteLine();
            DataFunctions.PrintData(featureLabels, targetLabels);
            DataFunctions.PrintData(featureData, targetData);



            Console.WriteLine();
            Console.WriteLine("Enter feature names to one-hot encode (type name and press enter or type 'DONE' and enter to stop):");
            List<string> featuresToOneHotEncode = new List<string>();

            {
                string userInput = "";

                while (userInput != "DONE")
                {
                    userInput = Console.ReadLine();
                    if (featuresToOneHotEncode.Contains(userInput))
                    {
                        continue;
                    }
                    else if (userInput == "DONE")
                    {
                        continue;
                    }
                    else if (!featureLabels[0].Contains(userInput))
                    {
                        continue;
                    }
                    else
                    {
                        featuresToOneHotEncode.Add(userInput);
                    }
                }
            }




            DataFunctions.HandleOneHotEncoding(featureData, featureLabels, featuresToOneHotEncode);

            Console.WriteLine();
            Console.WriteLine("Would you like to one-hot encode target data ('Y' or 'N'): ");

            {
                string userInput = "";

                while (userInput != "Y" && userInput != "N")
                {
                    userInput = Console.ReadLine();
                    if (userInput == "Y")
                    {
                        DataFunctions.HandleOneHotEncoding(targetData, targetLabels, targetLabels[0]);
                        break;
                    }
                    else if (userInput == "N")
                    {
                        break;
                    }
                    else
                    {

                    }
                }
            }

            Console.WriteLine();
            Console.WriteLine("--------------------");
            Console.WriteLine("One Hot Encoded Data");
            Console.WriteLine("--------------------");
            Console.WriteLine();

            Console.WriteLine("Features: " + (featureData[0].Count));
            Console.WriteLine("Targets: " + targetData[0].Count);
            Console.WriteLine("Examples: " + (featureData.Count));
            Console.WriteLine();

            DataFunctions.PrintData(featureLabels, targetLabels);
            DataFunctions.PrintData(featureData, targetData);



            double[,] totalInputData = DataFunctions.StringMatrixToDoubleMatrix(featureData);
            double[,] totalOutputData = DataFunctions.StringMatrixToDoubleMatrix(targetData);

            Console.WriteLine();
            Console.WriteLine("Enter feature names to normalize (type name or type 'DONE' and enter to stop): ");

            List<string> featuresToNormalize = new List<string>();

            {
                string userInput = "";

                while (userInput != "DONE")
                {
                    userInput = Console.ReadLine();
                    if (featuresToNormalize.Contains(userInput))
                    {
                        continue;
                    }
                    else if (userInput == "DONE")
                    {
                        continue;
                    }
                    else if (!featureLabels[0].Contains(userInput))
                    {
                        continue;
                    }
                    else
                    {
                        featuresToNormalize.Add(userInput);
                    }
                }
            }

            DataFunctions.HandleNormalization(totalInputData, featureLabels, featuresToNormalize);

            Console.WriteLine();
            Console.WriteLine("--------------------");
            Console.WriteLine("Normalized Data");
            Console.WriteLine("--------------------");
            Console.WriteLine();

            Console.WriteLine("Features: " + (featureData[0].Count));
            Console.WriteLine("Targets: " + targetData[0].Count);
            Console.WriteLine("Examples: " + (featureData.Count));
            Console.WriteLine();

            DataFunctions.PrintData(featureLabels, targetLabels);
            DataFunctions.PrintData(totalInputData, totalOutputData);

            Console.WriteLine();
            Console.WriteLine("Would you like to shuffle the data? ('Y' or 'N'): ");
            {
                string userInput = "";

                while (userInput != "Y" && userInput != "N")
                {
                    userInput = Console.ReadLine();
                    if (userInput == "Y")
                    {
                        DataFunctions.ShuffleRows(totalInputData, totalOutputData);

                        Console.WriteLine();
                        Console.WriteLine("-------------");
                        Console.WriteLine("Shuffled Data");
                        Console.WriteLine("-------------");
                        Console.WriteLine();

                        Console.WriteLine("Features: " + (featureData[0].Count));
                        Console.WriteLine("Targets: " + targetData[0].Count);
                        Console.WriteLine("Examples: " + (featureData.Count));
                        Console.WriteLine();

                        DataFunctions.PrintData(featureLabels, targetLabels);
                        DataFunctions.PrintData(totalInputData, totalOutputData);

                        break;
                    }
                    else if (userInput == "N")
                    {
                        break;
                    }
                    else
                    {

                    }
                }
            }

            Console.WriteLine();
            Console.WriteLine("Percentage of data for training (ie. '80' for eighty percent for training): ");

            {
                string userInput = "";
                bool goodInput = false;
                while (!goodInput)
                {
                    userInput = Console.ReadLine();
                    if (double.TryParse(userInput, out double number))
                    {
                        double percent = double.Parse(userInput);
                        int rowsForTrain = (int)(totalInputData.GetLength(0) * (percent / 100));
                        int rowsForTest = totalInputData.GetLength(0) - rowsForTrain;
                        featureTrainData = new double[rowsForTrain, totalInputData.GetLength(1)];
                        featureTestData = new double[rowsForTest, totalInputData.GetLength(1)];
                        targetTrainData = new double[rowsForTrain, totalOutputData.GetLength(1)];
                        targetTestData = new double[rowsForTest, totalOutputData.GetLength(1)];
                        DataFunctions.SplitData(totalInputData, featureTrainData, featureTestData, percent);
                        DataFunctions.SplitData(totalOutputData, targetTrainData, targetTestData, percent);
                        break;
                    }
                    else
                    {

                    }
                }
            }

            Console.WriteLine();
            Console.WriteLine("--------------------");
            Console.WriteLine("Split Data");
            Console.WriteLine("--------------------");
            Console.WriteLine();

            Console.WriteLine("Features: " + (featureData[0].Count));
            Console.WriteLine("Targets: " + targetData[0].Count);
            Console.WriteLine("Train Examples: " + (featureTrainData.GetLength(0)));
            Console.WriteLine("Test Examples: " + (featureTestData.GetLength(0)));
            Console.WriteLine();

            Console.WriteLine("----------");
            Console.WriteLine("Train Data");
            Console.WriteLine("----------");
            DataFunctions.PrintData(featureLabels, targetLabels);
            DataFunctions.PrintData(featureTrainData, targetTrainData);

            Console.WriteLine("---------");
            Console.WriteLine("Test Data");
            Console.WriteLine("---------");
            DataFunctions.PrintData(featureLabels, targetLabels);
            DataFunctions.PrintData(featureTestData, targetTestData);

            Console.WriteLine();
            Console.WriteLine("Saving data to CSV files");

            DataFunctions.CreateCSVFileInDirectory(pathToCSVDirectory, featureTrainData, featureTrainCSVName);
            DataFunctions.CreateCSVFileInDirectory(pathToCSVDirectory, featureTestData, featureTestCSVName);
            DataFunctions.CreateCSVFileInDirectory(pathToCSVDirectory, targetTrainData, targetTrainCSVName);
            DataFunctions.CreateCSVFileInDirectory(pathToCSVDirectory, targetTestData, targetTestCSVName);

        }
    }
}
