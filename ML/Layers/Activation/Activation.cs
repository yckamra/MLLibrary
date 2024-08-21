using System.Collections;
using System.Collections.Generic;
using System;

namespace ML
{

    public class Activation : Layer
    {
        protected Func<double[,], double[,]> activation;
        protected Func<double[,], double[,]> activationPrime;

        public Activation(Func<double[,], double[,]> activation, Func<double[,], double[,]> activationPrime)
        {
            this.activation = activation;
            this.activationPrime = activationPrime;
        }

        public override double[,] Forward(double[,] input, double[,] yTrue)
        {
            this.input = input;
            this.output = this.activation(input);
            return this.output;
        }

        public override double[,] Backward(double[,] outputGradient, double learningRate, int batchSize, Func<double, double[,], double[,], double[,], double[,], double[,], double[,], double[,]> OptimizationAlgorithm)
        {
            return NetworkFunctions.ElementWiseMultiplication(outputGradient, this.activationPrime(this.input));
        }

    }
}
