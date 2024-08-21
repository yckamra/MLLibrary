using System.Collections;
using System.Collections.Generic;
using System;

namespace ML
{

    public class Dense : Layer // TODO: Change Backward to make it more reusable and dynamic with batches and custom optimizers
    {
        protected double[,] weights;
        protected double[,] biases;
        protected int inputSize;
        protected int outputSize;
        protected double[,] weightsGradientCumulative;
        protected double[,] biasesGradientCumulative;
        protected int batchSizeCumulator;

        public Dense(int inputSize, int outputSize) // TESTED AND COMPLETE
        {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.weights = new double[outputSize, inputSize];
            this.biases = new double[outputSize, 1];
            this.batchSizeCumulator = 0;
            weightsGradientCumulative = new double[outputSize, inputSize];
            biasesGradientCumulative = new double[outputSize, 1];
            InitializeWeightsAndBiases();
            InitializeWeightsAndBiasesCumulative();
        }

        protected void InitializeWeightsAndBiases() // TESTED AND COMPLETE
        {
            for (int i = 0; i < outputSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    weights[i, j] = NetworkFunctions.RandomGaussianNumber();
                }
                biases[i, 0] = NetworkFunctions.RandomGaussianNumber();
            }
        }

        protected void InitializeWeightsAndBiasesCumulative() // TESTED AND COMPLETE
        {
            for (int i = 0; i < outputSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    weights[i, j] = 0;
                }
                biases[i, 0] = 0;
            }
        }

        public override double[,] Forward(double[,] input, double[,] yTrue) // TESTED AND COMPLETE
        {
            this.input = input;

            return NetworkFunctions.MatrixAddition(NetworkFunctions.DotProduct(weights, input), biases);
        }

        public override double[,] Backward(double[,] outputGradient, double learningRate, int batchSize, Func<double, double[,], double[,], double[,], double[,], double[,], double[,], double[,]> OptimizationAlgorithm)
        {
            ++batchSizeCumulator;
            double[,] Y = OptimizationAlgorithm(learningRate, outputGradient, input, weights, biases, weightsGradientCumulative, biasesGradientCumulative);

            if (batchSizeCumulator == batchSize)
            {
                weights = NetworkFunctions.MatrixSubtraction(weights, NetworkFunctions.ScalarMultiplication(NetworkFunctions.ScalarDivision(weightsGradientCumulative, batchSize), learningRate));
                biases = NetworkFunctions.MatrixSubtraction(biases, NetworkFunctions.ScalarMultiplication(NetworkFunctions.ScalarDivision(biasesGradientCumulative, batchSize), learningRate));

                InitializeWeightsAndBiasesCumulative();
                batchSizeCumulator = 0;
            }

            return Y;
        }

    }
}
