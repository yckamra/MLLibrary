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

        public static void OneHotEncodeColumnInPlace(List<List<string>> data, int columnIndex)
        {
            // Step 1: Find unique values in the specified column
            var uniqueValues = data.Select(row => row[columnIndex]).Distinct().ToList();

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

        public static void HandleOneHotEncoding(List<List<string>> data, List<string> featuresToOneHotEncode)
        {
            foreach(string feature in featuresToOneHotEncode)
            {
                int index = featuresToOneHotEncode.IndexOf(feature); // TODO: get new labels and remember that labels are different now
                OneHotEncodeColumnInPlace(data, index);
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

        public static void PrintList(List<List<string>> data) // TESTED AND COMPLETE
        {
            for (int i = 0; i < data.Count; i++)
            {
                string row = "";
                for (int j = 0; j < data[0].Count; j++)
                {
                    row += data[i][j];
                    if(j != data[0].Count - 1)
                    {
                        row += ", ";
                    }
                }
                Console.WriteLine("[" + row + "]");
            }
        }

        public static void HandleData(string filePath)
        {

            List<List<string>> data = new List<List<string>>();

            List<List<string>> labels = new List<List<string>>();
            List<string> label = new List<string>();
            labels.Add(label);

            CSVToStringList(data, filePath, ref labels);

            // TODO: get target feature this way we can make sure it is one hot encoded at the very end if need be

            Console.WriteLine("-----------");
            Console.WriteLine("Raw Dataset");
            Console.WriteLine("-----------");
            Console.WriteLine();
            Console.WriteLine("Features: " + (data[0].Count - 1));
            Console.WriteLine("Examples: " + (data.Count));
            Console.WriteLine();
            PrintList(labels);
            PrintList(data);

            Console.WriteLine();
            Console.WriteLine("Enter feature names to one-hot encode (type name and press enter or type DONE and enter to stop):");
            List<string> featuresToOneHotEncode = new List<string>();

            string userInput = "";

            while(userInput!= "DONE")
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
                else if (!labels[0].Contains(userInput))
                {
                    continue;
                }
                else
                {
                    featuresToOneHotEncode.Add(userInput);
                }
            }
            int sizeOfFeature = featuresToOneHotEncode.Count;
            HandleOneHotEncoding(data, featuresToOneHotEncode);

            Console.WriteLine();
            PrintList(data);

        }

        private static void CSVToStringList(List<List<string>> inputList, string filePath, ref List<List<string>> labels)
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
                            string valueAsString = value.ToString();
                            labels[0].Add(valueAsString);
                            column++;
                    }
                }
                
                int row = 0;
                // Read each line until the end of the file
                while (!reader.EndOfStream)
                {
                    int column = 0;
                    List<string> newLine = new List<string>();
                    inputList.Add(newLine);

                    // Read a line
                    var line = reader.ReadLine();

                    // Split the line into columns
                    var values = line.Split(',');

                    // Process the data
                    foreach (var value in values)
                    {
                            string valueAsString = value.ToString();
                            inputList[row].Add(valueAsString);
                            column++;
                    }
                    row++;
                }
            }
        }
    }
}
