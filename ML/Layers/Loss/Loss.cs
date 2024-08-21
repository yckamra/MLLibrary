using System.Collections;
using System.Collections.Generic;
using System;

namespace ML
{

    public class Loss : Layer // TODO: TESTING
    {

        protected Func<double[,], double[,], double[,]> loss;
        protected Func<double[,], double[,], double[,]> lossPrime;
        protected double[,] yTrue;

        public Loss(Func<double[,], double[,], double[,]> loss, Func<double[,], double[,], double[,]> lossPrime)
        {
            this.loss = loss;
            this.lossPrime = lossPrime;
        }

        // TODO: TEST
        public override double[,] Forward(double[,] input, double[,] yTrue)
        {
            this.yTrue = yTrue;
            this.input = input;
            this.output = loss(yTrue, input);
            return this.output;
        }

        // TODO: TEST
        public override double[,] Backward(double[,] outputGradient, double learningRate, int batchSize, Func<double, double[,], double[,], double[,], double[,], double[,], double[,], double[,]> OptimizationAlgorithm)
        {
            return lossPrime(this.yTrue, output);
        }
    }
}
