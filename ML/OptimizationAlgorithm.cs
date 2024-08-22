using System;
namespace ML
{
    public class OptimizationAlgorithm
    {

        // -----------------------
        // OPTIMIZATION ALGORITHMS
        // -----------------------

        public double[,] GradientDescent(double learningRate, double[,] outputGradient,
            double[,] input, double[,] weights, double[,] biases, ref double[,] weightsCumulative, ref double[,] biasesCumulative)
        {
            double[,] weightsGradient = NetworkFunctions.DotProduct(outputGradient, NetworkFunctions.Transpose(input));

            weightsCumulative = NetworkFunctions.MatrixAddition(weightsCumulative, weightsGradient);
            biasesCumulative = NetworkFunctions.MatrixAddition(biasesCumulative, outputGradient);

            return NetworkFunctions.DotProduct(NetworkFunctions.Transpose(weights), outputGradient);
        }

    }
}
