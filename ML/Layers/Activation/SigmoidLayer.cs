using System.Collections;
using System.Collections.Generic;
using System;

namespace ML
{
    public class SigmoidLayer : Activation
    {

        public SigmoidLayer() : base(Sigmoid, SigmoidPrime)
        {

        }

        private static double[,] Sigmoid(double[,] X)
        {
            int rows = X.GetLength(0);
            int columns = X.GetLength(1);

            double[,] Y = new double[rows, columns];

            for (int i = 0; i < rows; i++)
            {
                Y[i, 0] = 1 / (1 + Math.Exp(-1 * X[i, 0]));
            }

            return Y;
        }

        private static double[,] SigmoidPrime(double[,] X)
        {
            double[,] Y = NetworkFunctions.ElementWiseMultiplication(Sigmoid(X), OneMinusSigmoid(Sigmoid(X)));
            return Y;
        }

        private static double[,] OneMinusSigmoid(double[,] X)
        {
            int rows = X.GetLength(0);
            int columns = X.GetLength(1);

            double[,] Y = new double[rows, columns];

            for (int i = 0; i < rows; i++)
            {
                Y[i, 0] = 1 - X[i, 0];
            }
            return Y;
        }

    }
}
