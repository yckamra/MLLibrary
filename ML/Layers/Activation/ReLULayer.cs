using System.Collections;
using System.Collections.Generic;
using System;

namespace ML
{

    public class ReLULayer : Activation
    {

        public ReLULayer() : base(ReLU, ReLUPrime)
        {

        }

        private static double[,] ReLU(double[,] X)
        {
            int rows = X.GetLength(0);
            int columns = X.GetLength(1);

            double[,] Y = new double[rows, columns];

            for (int i = 0; i < rows; i++)
            {

                if (X[i, 0] > 0f)
                {
                    Y[i, 0] = X[i, 0];
                }
                else
                {
                    Y[i, 0] = 0;
                }
            }
            return Y;
        }

        private static double[,] ReLUPrime(double[,] X)
        {
            int rows = X.GetLength(0);
            int columns = X.GetLength(1);

            double[,] Y = new double[rows, columns];

            for (int i = 0; i < rows; i++)
            {
                if (X[i, 0] > 0)
                {
                    Y[i, 0] = 1;
                }
                else
                {
                    Y[i, 0] = 0;
                }
            }
            return Y;
        }
    }
}
