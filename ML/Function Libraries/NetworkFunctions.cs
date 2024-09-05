using System.Collections;
using System.Collections.Generic;
using System;
using System.IO;
using System.Linq;

namespace ML // TODO: Encapsulate, organize, and error check; all Handle functions must go to new class file
{
    public static class NetworkFunctions
    {
        // ------------------------
        // RANDOM NUMBER GENERATION
        // ------------------------

        private static System.Random rand = new System.Random();

        public static double RandomGaussianNumber(double mean, double stdDev) // TESTED AND COMPLETE
        {
            double randNormal = 0;

            double u1 = 1.0 - rand.NextDouble();
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                       Math.Sin(2.0 * Math.PI * u2);
            randNormal = mean + stdDev * randStdNormal;

            return randNormal;
        }

        public static double HeInitialization(int rows) // input values
        {
            double standardDeviation = Math.Sqrt(2.0 / rows);

            return RandomGaussianNumber(0, standardDeviation);
        }

        public static double XavierInitialization(int rows, int columns)
        {
            double limit = Math.Sqrt(6.0 / (rows + columns));
            return (2 * limit * rand.NextDouble()) - limit;
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
    }
}
