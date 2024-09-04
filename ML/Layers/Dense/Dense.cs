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
        protected int iteration;

        public Dense(int inputSize, int outputSize) // TESTED AND COMPLETE
        {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.weights = new double[outputSize, inputSize];
            this.biases = new double[outputSize, 1];
            weightsGradientCumulative = new double[outputSize, inputSize];
            biasesGradientCumulative = new double[outputSize, 1];
            InitializeWeightsAndBiases();
            InitializeWeightsAndBiasesCumulative();
            iteration = 0;
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
                    weightsGradientCumulative[i, j] = 0;
                }
                biasesGradientCumulative[i, 0] = 0;
            }
        }

        public override double[,] Forward(double[,] input, double[,] yTrue) // TESTED AND COMPLETE
        {
            this.input = input;

            double[,] Y = NetworkFunctions.MatrixAddition(NetworkFunctions.DotProduct(weights, input), biases);

            return Y;
        }

        public override double[,] Backward(double[,] outputGradient, double learningRate, int batchSize,
            OptimizationAlgorithm optimization)
        {
            double[,] Y = optimization.GradientDescent(learningRate, outputGradient, input, weights, biases,
                ref weightsGradientCumulative, ref biasesGradientCumulative);

            iteration++;

            if (iteration == batchSize) {
                weights = NetworkFunctions.MatrixSubtraction(weights,
                    NetworkFunctions.ScalarMultiplication(NetworkFunctions.ScalarDivision(weightsGradientCumulative,
                    batchSize), learningRate));
                biases = NetworkFunctions.MatrixSubtraction(biases,
                    NetworkFunctions.ScalarMultiplication(NetworkFunctions.ScalarDivision(biasesGradientCumulative,
                    batchSize), learningRate));

                InitializeWeightsAndBiasesCumulative();
                iteration = 0;
            }

            return Y;
        }
    }
}
