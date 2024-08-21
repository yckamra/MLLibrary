using System.Collections;
using System.Collections.Generic;
using System;

namespace ML
{
    public class BinaryCrossEntropyLayer : Loss // TODO: NOT DONE
    {
        public BinaryCrossEntropyLayer() : base(BinaryCrossEntropy, BinaryCrossEntropyPrime) // DONE
        {

        }

        private static double[,] BinaryCrossEntropy(double[,] yTrue, double[,] yPredicted) // TODO: NOT DONE
        {
            double[,] Y = new double[1, 1];
            Y[0, 0] = -1 * ((yTrue[0, 0] * Math.Log(yPredicted[0, 0])) + ((1 - yTrue[0, 0]) * Math.Log(1 - yPredicted[0, 0])));

            return Y;
        }

        private static double[,] BinaryCrossEntropyPrime(double[,] yTrue, double[,] yPredicted) // TODO: NOT DONE
        {
            double[,] Y = new double[1, 1];
            Y[0, 0] = (-1 * (yTrue[0, 0] / yPredicted[0, 0])) + ((1 - yTrue[0, 0]) / (1 - yPredicted[0, 0]));

            return Y;
        }
    }
}
