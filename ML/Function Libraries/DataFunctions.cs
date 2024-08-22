using System.Collections;
using System.Collections.Generic;
using System;
using System.IO;
using System.Linq;

namespace ML
{
    public static class DataFunctions
    {
        // ----------------
        // DATASET HANDLING
        // ----------------
        // TODO: the data is expected to have target in last column but we should check this for more functionality and control

        // CSV -> List<List<string>> for all features and List<List<string>> for targets -> one hot encode features ->
        // one hot encode targets if needed -> turn to double[,] for both features and targets -> normalize features ->
        // shuffle? -> split

        // This allows for adding features to the end of the dataset matrix
        public static void AppendMatrix(List<List<string>> originalMatrix, List<List<string>> matrixToAppend)
        {
            for (int i = 0; i < matrixToAppend.Count; i++)
            {
                for (int j = 0; j < matrixToAppend[i].Count; j++)
                {
                    originalMatrix[i].Add(matrixToAppend[i][j]);
                }
            }
        }

        public static double[,] StringMatrixToDoubleMatrix(List<List<string>> stringMatrix)
        {
            int rows = stringMatrix.Count;
            int columns = stringMatrix[0].Count;
            double[,] Y = new double[rows, columns];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    Y[i, j] = Double.Parse(stringMatrix[i][j]);
                }
            }
            return Y;
        }

        public static void OneHotEncodeColumnInPlace(List<List<string>> data, int columnIndex, List<List<string>> labels)
        {
            // Step 1: Find unique values in the specified column
            var uniqueValues = data.Select(row => row[columnIndex]).Distinct().ToList();

            List<string> newLabels = new List<string>();
            foreach (var value in uniqueValues)
            {
                newLabels.Add(value);
            }
            labels[0].RemoveAt(columnIndex);
            labels[0].InsertRange(columnIndex, newLabels);
            // Step 2: Modify the data in place
            foreach (var row in data)
            {
                // Step 2a: Create new one-hot encoded values for the column
                List<string> oneHotEncodedValues = uniqueValues
                    .Select(value => row[columnIndex] == value ? "1" : "0")
                    .ToList();

                // Step 2b: Remove the original column value
                row.RemoveAt(columnIndex);

                // Step 2c: Insert the new one-hot encoded columns in place of the original column
                row.InsertRange(columnIndex, oneHotEncodedValues);
            }
        }

        public static void HandleOneHotEncoding(List<List<string>> data, List<List<string>> labels, List<string> featuresToOneHotEncode)
        {
            int count = featuresToOneHotEncode.Count;
            for (int i = 0; i < count; i++)
            {
                string feature = featuresToOneHotEncode[i];
                int index = labels[0].IndexOf(feature);
                OneHotEncodeColumnInPlace(data, index, labels);
            }
        }

        public static void HandleNormalization(double[,] data, List<List<string>> labels, List<string> featuresToNormalize)
        {
            int count = featuresToNormalize.Count;
            for (int i = 0; i < count; i++)
            {
                string feature = featuresToNormalize[i];
                int index = labels[0].IndexOf(feature);
                NormalizeFeatureColumn(data, index);
            }
        }

        public static void NormalizeFeatureColumn(double[,] inputMatrix, int featureColumnNumber)
        {
            int column = featureColumnNumber;
            double[,] inputFeatureColumn = NetworkFunctions.GetAColumn(inputMatrix, column);
            int rows = inputFeatureColumn.GetLength(0);
            double mean = NetworkFunctions.Mean(inputFeatureColumn);
            double standardDeviation = NetworkFunctions.StandardDeviation(inputFeatureColumn, mean);
            for (int i = 0; i < rows; i++)
            {
                inputMatrix[i, column] = (inputMatrix[i, column] - mean) / standardDeviation;
            }
        }

        public static void SplitData(double[,] totalData, double[,] trainData, double[,]
    crossValidateData, double percentForTrain)
        {

            int totalExamples = totalData.GetLength(0);
            int examplesForTrain = (int)(totalExamples * (percentForTrain / 100));
            int examplesForCrossValidate = totalExamples - examplesForTrain;
            int columns = totalData.GetLength(1);


            for (int i = 0; i < examplesForTrain; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    trainData[i, j] = totalData[i, j];
                }
            }

            for (int i = 0; i < examplesForCrossValidate; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    crossValidateData[i, j] = totalData[examplesForTrain + i, j];
                }
            }
        }

        public static void ShuffleRows(double[,] inputArray, double[,] outputArray)
        {
            int rowCount = inputArray.GetLength(0);
            Random random = new Random();

            for (int i = rowCount - 1; i > 0; i--)
            {
                // Pick a random index from 0 to i
                int j = random.Next(i + 1);

                // Swap the rows
                SwapRows(inputArray, i, j);
                SwapRows(outputArray, i, j);
            }
        }

        public static void SwapRows(double[,] array, int row1, int row2)
        {
            int columnCount = array.GetLength(1);
            for (int col = 0; col < columnCount; col++)
            {
                double temp = array[row1, col];
                array[row1, col] = array[row2, col];
                array[row2, col] = temp;
            }
        }

        public static void PrintData(List<List<string>> featureData, List<List<string>> targetData) // TESTED AND COMPLETE
        {
            for (int i = 0; i < featureData.Count; i++)
            {
                string row = "";
                for (int j = 0; j < featureData[0].Count; j++)
                {
                    row += featureData[i][j];
                    if (j != featureData[0].Count - 1)
                    {
                        row += ", ";
                    }
                }
                Console.Write("[" + row + "]");

                string targetRow = "";
                for (int j = 0; j < targetData[0].Count; j++)
                {
                    targetRow += targetData[i][j];
                    if (j != targetData[0].Count - 1)
                    {
                        targetRow += ", ";
                    }
                }

                Console.WriteLine("[" + targetRow + "]");
            }
        }

        public static void PrintData(double[,] featureData, double[,] targetData) // TESTED AND COMPLETE
        {
            for (int i = 0; i < featureData.GetLength(0); i++)
            {
                string row = "";
                for (int j = 0; j < featureData.GetLength(1); j++)
                {
                    row += featureData[i, j];
                    if (j != featureData.GetLength(1) - 1)
                    {
                        row += ", ";
                    }
                }
                Console.Write("[" + row + "]");

                string targetRow = "";
                for (int j = 0; j < targetData.GetLength(1); j++)
                {
                    targetRow += targetData[i, j];
                    if (j != targetData.GetLength(1) - 1)
                    {
                        targetRow += ", ";
                    }
                }

                Console.WriteLine("[" + targetRow + "]");
            }
        }

        public static void CSVToStringList(List<List<string>> featureData, List<List<string>> featureLabels,
    List<List<string>> targetData, List<List<string>> targetLabels, string filePath)
        {
            // Open the file using StreamReader
            using (var reader = new StreamReader(filePath))
            {
                {

                    int column = 0;
                    // Read a line
                    var line = reader.ReadLine();

                    // Split the line into columns
                    var values = line.Split(',');

                    // Process the data
                    foreach (var value in values)
                    {
                        if (column != values.Length - 1)
                        {
                            string valueAsString = value.ToString();
                            featureLabels[0].Add(valueAsString);
                            column++;
                        }
                        else
                        {
                            string valueAsString = value.ToString();
                            targetLabels[0].Add(valueAsString);
                            column++;
                        }
                    }
                }

                int row = 0;
                // Read each line until the end of the file
                while (!reader.EndOfStream)
                {
                    List<string> targetLine = new List<string>();
                    targetData.Add(targetLine);
                    List<string> dataLine = new List<string>();
                    featureData.Add(dataLine);

                    int column = 0;

                    // Read a line
                    var line = reader.ReadLine();

                    // Split the line into columns
                    var values = line.Split(',');

                    // Process the data
                    foreach (var value in values)
                    {
                        if (column != values.Length - 1)
                        {

                            string valueAsString = value.ToString();
                            featureData[row].Add(valueAsString);
                            column++;
                        }
                        else
                        {

                            string valueAsString = value.ToString();
                            targetData[row].Add(valueAsString);
                            column++;
                        }
                    }
                    row++;
                }
            }
        }

        public static void LoadCSV(List<List<string>> data, string filePath)
        {
            // Open the file using StreamReader
            using (var reader = new StreamReader(filePath))
            {

                int row = 0;
                // Read each line until the end of the file
                while (!reader.EndOfStream)
                {
                    List<string> newRow = new List<string>();
                    data.Add(newRow);

                    int column = 0;

                    // Read a line
                    var line = reader.ReadLine();

                    // Split the line into columns
                    var values = line.Split(',');

                    // Process the data
                    foreach (var value in values)
                    {
                        string valueAsString = value.ToString();
                        data[row].Add(valueAsString);
                        column++;
                    }
                    row++;
                }
            }
        }

        public static void CreateCSVFileInDirectory(string directoryPath, double[,] data, string fileName)
        {
            // Ensure the directory exists, create it if it doesn't
            if (!Directory.Exists(directoryPath))
            {
                Directory.CreateDirectory(directoryPath);
            }

            // Combine directory path and file name to get the full file path
            string filePath = Path.Combine(directoryPath, fileName);

            // Create and fill the CSV file
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                int rows = data.GetLength(0);
                int cols = data.GetLength(1);

                for (int i = 0; i < rows; i++)
                {
                    string[] rowValues = new string[cols];

                    for (int j = 0; j < cols; j++)
                    {
                        // Convert double values to string
                        rowValues[j] = data[i, j].ToString();
                    }

                    // Join the values with commas and write as a line in the CSV file
                    string row = string.Join(",", rowValues);
                    writer.WriteLine(row);
                }
            }

            Console.WriteLine($"CSV file saved at: {filePath}");
        }
    }
}
