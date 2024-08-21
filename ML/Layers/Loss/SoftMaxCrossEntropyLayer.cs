using System.Collections;
using System.Collections.Generic;
using System;

namespace ML
{

    public class SoftMaxCrossEntropyLayer : Loss // TODO: NOT DONE
    {
        public SoftMaxCrossEntropyLayer() : base(SoftMaxCrossEntropy, SoftMaxCrossEntropyPrime) // DONE
        {

        }

        // yPredicted and yTrue are expected to be one hot encoded for a single training example
        private static double[,] SoftMaxCrossEntropy(double[,] yTrue, double[,] yPredicted) // TODO: NOT DONE
        {
            int rows = yTrue.GetLength(0);
            double[,] Y = new double[1, 1];
            Y[0, 0] = 0;

            for (int i = 0; i < rows; i++)
            {
                Y[0, 0] += (yTrue[i, 0] * Math.Log(yPredicted[i, 0]));
            }
            Y[0, 0] *= -1;

            return Y;
        }

        private static double[,] SoftMaxCrossEntropyPrime(double[,] yTrue, double[,] yPredicted) // TODO: NOT DONE
        {
            int rows = yTrue.GetLength(0);
            double[,] Y = new double[rows, 1];
            for (int i = 0; i < rows; i++)
            {
                Y[i, 0] = yPredicted[i, 0] - yTrue[i, 0];
            }

            return Y;
        }
    }
}
