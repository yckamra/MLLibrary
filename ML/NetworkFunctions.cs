using System.Collections;
using System.Collections.Generic;
using System;
using System.IO;
using System.Linq;

namespace ML // TODO: Encapsulate, organize, and error check
{
    public static class NetworkFunctions
    {
        // -----------------------
        // OPTIMIZATION ALGORITHMS
        // -----------------------

        public static double[,] GradientDescent(double learningRate, double[,] outputGradient,
            double[,] input, double[,] weights, double[,] biases, double[,] weightsCumulative, double[,] biasesCumulative)
        {
            double[,] weightsGradient = NetworkFunctions.DotProduct(outputGradient, NetworkFunctions.Transpose(input));

            weightsCumulative = NetworkFunctions.MatrixAddition(weightsCumulative, weightsGradient);
            biasesCumulative = NetworkFunctions.MatrixAddition(biasesCumulative, outputGradient);

            return NetworkFunctions.DotProduct(NetworkFunctions.Transpose(weights), outputGradient);
        }





        // ------------------------
        // RANDOM NUMBER GENERATION
        // ------------------------

        private static System.Random rand = new System.Random();

        public static double RandomGaussianNumber(double mean = 0, double stdDev = 1) // TESTED AND COMPLETE
        {
            double randNormal = 0;

            double u1 = 1.0 - rand.NextDouble();
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                       Math.Sin(2.0 * Math.PI * u2);
            randNormal = mean + stdDev * randStdNormal;

            return randNormal;
        }





        // -----------------------
        // MATHEMATICAL OPERATIONS
        // -----------------------

        public static double Power(double X, int powerOf) // TESTED AND COMPLETE
        {
            double Y = 1;

            for (int i = 0; i < powerOf; i++)
            {
                Y *= X;
            }

            return Y;
        }

        public static double Mean(double[,] featureMatrix) // TESTED AND COMPLETE
        {
            int rows = featureMatrix.GetLength(0);
            double mean = 0;
            for (int i = 0; i < rows; i++)
            {
                mean += featureMatrix[i, 0];
            }
            mean /= rows;

            return mean;
        }

        public static double StandardDeviation(double[,] featureMatrix, double mean) // TESTED AND COMPLETE
        {
            int rows = featureMatrix.GetLength(0);
            double standardDeviation = 0;

            for (int i = 0; i < rows; i++)
            {
                standardDeviation += NetworkFunctions.Power((featureMatrix[i, 0] - mean), 2);
            }
            standardDeviation /= rows;

            return standardDeviation;
        }





        // ---------------------------
        // MATRIX TO MATRIX OPERATIONS
        // ---------------------------

        // double[i, j] where i is the vertical for rows and j is the horizontal for columns
        public static double[,] DotProduct(double[,] matrixA, double[,] matrixB) // TESTED AND COMPLETE
        {

            int rowsA = matrixA.GetLength(0);
            int colsA = matrixA.GetLength(1);
            int rowsB = matrixB.GetLength(0);
            int colsB = matrixB.GetLength(1);


            double[,] returnMatrix = new double[rowsA, colsB];

            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < colsB; j++)
                {
                    returnMatrix[i, j] = 0;
                    for (int k = 0; k < colsA; k++)
                    {
                        returnMatrix[i, j] += matrixA[i, k] * matrixB[k, j];
                    }
                }
            }

            return returnMatrix;
        }

        public static double[,] MatrixAddition(double[,] matrixA, double[,] matrixB) // TESTED AND COMPLETE
        {
            double[,] returnMatrix = new double[matrixA.GetLength(0), matrixA.GetLength(1)];

            for (int i = 0; i < matrixA.GetLength(0); i++) // row
            {
                for (int j = 0; j < matrixA.GetLength(1); j++) // column
                {
                    returnMatrix[i, j] = matrixA[i, j] + matrixB[i, j];
                }
            }

            return returnMatrix;
        }

        public static double[,] MatrixSubtraction(double[,] matrixA, double[,] matrixB) // TESTED AND COMPLETE
        {
            double[,] returnMatrix = new double[matrixA.GetLength(0), matrixA.GetLength(1)];

            for (int i = 0; i < matrixA.GetLength(0); i++) // row
            {
                for (int j = 0; j < matrixA.GetLength(1); j++) // column
                {
                    returnMatrix[i, j] = matrixA[i, j] - matrixB[i, j];
                }
            }

            return returnMatrix;
        }

        public static double[,] ElementWiseMultiplication(double[,] matrixA, double[,] matrixB) // TESTED AND COMPLETE
        {
            int rows = matrixA.GetLength(0);
            int columns = matrixA.GetLength(1);

            double[,] Y = new double[rows, columns];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    Y[i, j] = matrixA[i, j] * matrixB[i, j];
                }
            }

            return Y;
        }





        // ---------------------------
        // MATRIX TO SCALAR OPERATIONS
        // ---------------------------

        public static double[,] ScalarMultiplication(double[,] matrix, double scalar) // TESTED AND COMPLETE
        {
            double[,] returnMatrix = new double[matrix.GetLength(0), matrix.GetLength(1)];

            for (int i = 0; i < matrix.GetLength(0); i++) // row
            {
                for (int j = 0; j < matrix.GetLength(1); j++) // column
                {
                    returnMatrix[i, j] = matrix[i, j] * scalar;
                }
            }

            return returnMatrix;
        }

        public static double[,] ScalarDivision(double[,] matrix, double scalar) // TESTED AND COMPLETE
        {
            double[,] returnMatrix = new double[matrix.GetLength(0), matrix.GetLength(1)];

            for (int i = 0; i < matrix.GetLength(0); i++) // row
            {
                for (int j = 0; j < matrix.GetLength(1); j++) // column
                {
                    returnMatrix[i, j] = matrix[i, j] / scalar;
                }
            }

            return returnMatrix;
        }

        public static double[,] ScalarMatrixAddition(double[,] X, double scalar) // For subtraction just pass negative scalar
        {
            int rows = X.GetLength(0);
            int columns = X.GetLength(1);

            double[,] Y = new double[rows, columns];

            for (int i = 0; i < rows; i++)
            {
                Y[i, 0] = scalar + X[i, 0];
            }
            return Y;
        }





        // -----------------------------------
        // MATRIX MANIPULATION AND ALTERATION
        // -----------------------------------

        public static double[,] Transpose(double[,] matrix) // TESTED AND COMPLETE
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            double[,] transposedMatrix = new double[columns, rows];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    transposedMatrix[j, i] = matrix[i, j];
                }
            }

            return transposedMatrix;
        }

        public static double[,] GetARow(double[,] matrix, int rowIndex) // TESTED AND COMPLETE
        {
            int rowLength = matrix.GetLength(1);
            double[,] Y = new double[1, rowLength];

            for (int i = 0; i < rowLength; i++)
            {
                Y[0, i] = matrix[rowIndex, i];
            }

            return Y;
        }

        public static double[,] GetAColumn(double[,] matrix, int columnIndex) // TESTED AND COMPLETE
        {
            int columnLength = matrix.GetLength(0);
            double[,] Y = new double[columnLength, 1];

            for (int i = 0; i < columnLength; i++)
            {
                Y[i, 0] = matrix[i, columnIndex];
            }

            return Y;
        }

        public static void PrintMatrix(double[,] weights) // TESTED AND COMPLETE
        {
            int rows = weights.GetLength(0);
            int columns = weights.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    Console.WriteLine("Row: " + i + " Column: " + j + " = " + weights[i, j]);
                }
            }
        }

        public static void MatrixHardCopy(double[,] TheThingMatrix, double[,] MatrixToCopy)
        {
            int rows = MatrixToCopy.GetLength(0);
            int columns = MatrixToCopy.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    TheThingMatrix[i, j] = MatrixToCopy[i, j];
                }
            }
        }






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
            foreach(var value in uniqueValues)
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
            for(int i = 0; i < count; i++)
            {
                string feature = featuresToOneHotEncode[i];
                int index = labels[0].IndexOf(feature);
                OneHotEncodeColumnInPlace(data, index, labels);
            }
        }

        public static void HandleNormalization(double[,] data, List<List<string>> labels, List<string> featuresToNormalize)
        {
            int count = featuresToNormalize.Count;
            for(int i = 0; i < count; i++)
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

        public static void SplitData(double[,] totalData, double[,] trainDataInput, double[,]
    crossValidateDataInput, double[,] trainDataOutput, double[,] crossValidateDataOutput, double percentForTrain)
        {

            int totalExamples = totalData.GetLength(0);
            int examplesForTrain = (int)(totalExamples * percentForTrain);
            int examplesForCrossValidate = totalExamples - examplesForTrain;
            int columns = totalData.GetLength(1);


            ShuffleRows(totalData); // TODO: ask user if shuffle (with big data sets this will be inefficient)

            for (int i = 0; i < examplesForTrain; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    trainDataInput[i, j] = totalData[i, j];
                }
            }

            for (int i = 0; i < examplesForCrossValidate; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    crossValidateDataInput[i, j] = totalData[examplesForTrain + i, j];
                }
            }
        }

        private static void ShuffleRows(double[,] inputArray)
        {
            int rowCount = inputArray.GetLength(0);
            Random random = new Random();

            for (int i = rowCount - 1; i > 0; i--)
            {
                // Pick a random index from 0 to i
                int j = random.Next(i + 1);

                // Swap the rows
                SwapRows(inputArray, i, j);
            }
        }

        private static void SwapRows(double[,] array, int row1, int row2)
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
                    if(j != featureData[0].Count - 1)
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
                    row += featureData[i,j];
                    if (j != featureData.GetLength(1) - 1)
                    {
                        row += ", ";
                    }
                }
                Console.Write("[" + row + "]");

                string targetRow = "";
                for (int j = 0; j < targetData.GetLength(1); j++)
                {
                    targetRow += targetData[i,j];
                    if (j != targetData.GetLength(1) - 1)
                    {
                        targetRow += ", ";
                    }
                }

                Console.WriteLine("[" + targetRow + "]");
            }
        }

        private static void CSVToStringList(List<List<string>> featureData, List<List<string>> featureLabels,
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

        public static void HandleData(string filePath)
        {

            List<List<string>> featureData = new List<List<string>>();
            List<List<string>> targetData = new List<List<string>>();
            List<List<string>> featureLabels = new List<List<string>>();
            List<List<string>> targetLabels = new List<List<string>>();

            List<string> featureLabel = new List<string>();
            featureLabels.Add(featureLabel);
            List<string> targetLabel = new List<string>();
            targetLabels.Add(targetLabel);


            CSVToStringList(featureData, featureLabels, targetData, targetLabels, filePath);


            Console.WriteLine("-----------");
            Console.WriteLine("Raw Dataset");
            Console.WriteLine("-----------");
            Console.WriteLine();
            Console.WriteLine("Features: " + (featureData[0].Count));
            Console.WriteLine("Targets: " + targetData[0].Count);
            Console.WriteLine("Examples: " + (featureData.Count));
            Console.WriteLine();
            PrintData(featureLabels, targetLabels);
            PrintData(featureData, targetData);

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
            
            HandleOneHotEncoding(featureData, featureLabels, featuresToOneHotEncode);

            Console.WriteLine();
            Console.WriteLine("Would you like to one-hot encode target data ('Y' or 'N'): ");

            {
                string userInput = "";

                while (userInput != "Y" && userInput != "N")
                {
                    userInput = Console.ReadLine();
                    if (userInput == "Y")
                    {
                        HandleOneHotEncoding(targetData, targetLabels, targetLabels[0]);
                        break;
                    }else if (userInput == "N")
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

            PrintData(featureLabels, targetLabels);
            PrintData(featureData, targetData);



            double[,] totalInputData = StringMatrixToDoubleMatrix(featureData);
            double[,] totalOutputData = StringMatrixToDoubleMatrix(targetData);

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

            HandleNormalization(totalInputData, featureLabels, featuresToNormalize);

            Console.WriteLine();
            Console.WriteLine("--------------------");
            Console.WriteLine("Normalized Data");
            Console.WriteLine("--------------------");
            Console.WriteLine();

            Console.WriteLine("Features: " + (featureData[0].Count));
            Console.WriteLine("Targets: " + targetData[0].Count);
            Console.WriteLine("Examples: " + (featureData.Count));
            Console.WriteLine();

            PrintData(featureLabels, targetLabels);
            PrintData(totalInputData, totalOutputData);

        }

    }
}
