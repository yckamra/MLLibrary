using System.Collections;
using System.Collections.Generic;
using System;

namespace ML
{

    public class Layer
    {
        protected double[,] input;
        protected double[,] output;

        public virtual double[,] Forward(double[,] input, double[,] yTrue)
        {
            return null;
        }

        public virtual double[,] Backward(double[,] outputGradient, double learningRate, int batchSize, Func<double, double[,], double[,], double[,], double[,], double[,], double[,], double[,]> OptimizationAlgorithm)
        {
            return null;
        }

    }
}
