using System.Collections;
using System.Collections.Generic;
using System;

namespace ML
{

    public class SoftMaxLayer : Activation // Use this layer with a SoftMaxCrossEntropyLayer for the loss function
    {

        public SoftMaxLayer() : base(SoftMax, SoftMaxPrime)
        {


        }

        private static double[,] SoftMax(double[,] X)
        {
            int rows = X.GetLength(0);
            int columns = X.GetLength(1);
            double[,] Y = new double[rows, columns];
            double denominatorSum = 0;
            for (int i = 0; i < rows; i++)
            {
                Y[i, 0] = Math.Exp(X[i, 0]);
                denominatorSum += Y[i, 0];
            }
            for (int i = 0; i < Y.GetLength(0); i++)
            {
                Y[i, 0] = Y[i, 0] / denominatorSum;
            }

            return Y;
        }

        private static double[,] SoftMaxPrime(double[,] X)
        {
            int rows = X.GetLength(0);
            double[,] Y = new double[rows, 1];

            for (int i = 0; i < rows; i++)
            {
                Y[i, 1] = 1;
            }

            return Y;
        }
    }
}
