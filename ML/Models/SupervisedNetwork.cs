using System.Collections;
using System.Collections.Generic;
using System;

namespace ML
{
    // TODO: make this class SupervisedNetwork which is child of purely virtual parent Network
    public class SupervisedNetwork
    {
        // Layers includes: Dense layers, Activation layers, AND the Loss layer
        // so a full forward pass which includes the Loss layer will give the loss
        // of an example and not the prediction of the network.
        public List<Layer> layers;

        private void ForwardPass(ref double[,] inputForNextLayer, double[,] yTrueForExample)
        {
            foreach (Layer layer in this.layers)
            {
                if (layer is Dense childDense)
                {
                    inputForNextLayer = childDense.Forward(inputForNextLayer, yTrueForExample);

                    continue;
                }
                if (layer is Activation childActivation)
                {
                    inputForNextLayer = childActivation.Forward(inputForNextLayer, yTrueForExample);

                    continue;
                }
                if (layer is Loss childLoss) // naming here is a little dark
                {
                    inputForNextLayer = childLoss.Forward(inputForNextLayer, yTrueForExample);

                    continue;
                }
            }
        }

        private void BackwardPass(ref double[,] outputGradient, double learningRate, int batchSize,
            OptimizationAlgorithm optimization)
        {
            foreach (Layer layer in this.layers)
            {
                if (layer is Dense childDense)
                {
                    outputGradient = childDense.Backward(outputGradient, learningRate, batchSize, optimization);

                    continue;
                }
                if (layer is Activation childActivation)
                {
                    outputGradient = childActivation.Backward(outputGradient, learningRate, batchSize, optimization);

                    continue;
                }
                if (layer is Loss childLoss)
                {
                    outputGradient = childLoss.Backward(outputGradient, learningRate, batchSize, optimization);

                    continue;
                }
            }
        }

        // The input should have a row as a single training example
        public void Train(double[,] input, double[,] yTrue, int epochs, double learningRate, int batchSize,
            OptimizationAlgorithm optimization, double[,] featureTestData, double[,] targetTestData) // learning rate, output gradient, input, weights, biases, weightsCumulative, biasesCumulative, outputs a double[,]
        {
            for (int i = 0; i < epochs; i++)
            {
                Console.WriteLine("Epoch: " + i);

                double batchCost = 0; // cost of batch
                int batchIteration = 0;

                double epochCost = 0;
                int numberOfBatches = 0;

                int trainingExamples = yTrue.GetLength(0); // # rows (which is the # of training examples)
                int numberOfLayers = layers.Count; // Hidden layers, output layer, and loss layer included

                for (int j = 0; j < trainingExamples; j++)
                {
                    double[,] inputForNextLayer = NetworkFunctions.Transpose(NetworkFunctions.GetARow(input, j)); // transpose the training example to be vertical for input
                    double[,] yTrueForExample = new double[1, 1];
                    yTrueForExample[0, 0] = yTrue[j, 0]; // the particular yTrue for the training example

                    ForwardPass(ref inputForNextLayer, yTrueForExample);

                    batchCost += inputForNextLayer[0,0]; // cost of epoch
                    batchIteration++; // Increment examples in batch

                    if(batchIteration == batchSize)
                    {
                        batchCost /= batchIteration;
                        // Console.WriteLine("Batch Cost: " + batchCost);

                        epochCost += batchCost;
                        numberOfBatches++;

                        batchCost = 0;
                        batchIteration = 0;
                    }

                    layers.Reverse();

                    // We want to train all examples so if the batch size is bigger than the number of examples left
                    // we set the batch size to the remainder

                    double[,] outputGradient = null;

                    BackwardPass(ref outputGradient, learningRate, batchSize, optimization);

                    layers.Reverse();
                }
                epochCost /= numberOfBatches;
                Console.WriteLine("Epoch Cost: " + epochCost);

                double trainAccuracy = DataFunctions.TrainAccuracy(this, input, yTrue); // no touch
                double testAccuracy = DataFunctions.TestAccuracy(this, featureTestData, targetTestData); // no touch

                Console.WriteLine("Train Accuracy: " + trainAccuracy); // no touch
                Console.WriteLine("Test Accuracy: " + testAccuracy); // no touch
            }
        }

        public double[,] Predict(double[,] input)
        {
            double[,] inputForNextLayer = NetworkFunctions.Transpose(input);
            foreach (Layer layer in this.layers)
            {
                if (layer is Loss childLoss)
                {
                    continue;
                }
                else
                {
                    inputForNextLayer = layer.Forward(inputForNextLayer, null);
                }
            }
            return inputForNextLayer;
        }
    }
}
