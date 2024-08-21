using System.Collections;
using System.Collections.Generic;
using System;

namespace ML
{

    public class SquaredErrorLayer : Loss // TODO: NOT DONE
    {
        public SquaredErrorLayer() : base(SquaredError, SquaredErrorPrime) // DONE
        {

        }
        private static double[,] SquaredError(double[,] yTrue, double[,] yPredicted) // for SGD
        {
            double[,] Y = new double[1, 1];
            Y[0, 0] = NetworkFunctions.Power(yTrue[0, 0] - yPredicted[0, 0], 2);
            return Y;
        }

        private static double[,] SquaredErrorPrime(double[,] yTrue, double[,] yPredicted)
        {
            double[,] Y = new double[1, 1];
            Y[0, 0] = 2 * (yPredicted[0, 0] - yTrue[0, 0]);

            return Y;
        }

    }
}
