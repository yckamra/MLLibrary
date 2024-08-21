using System.Collections;
using System.Collections.Generic;
using System;
using System.IO;

namespace ML
{
    public static class UIandDriver
    {
        public static void SupervisedLearningUI()
        {
            // (1) Create a filePath

            string filePath = "Assets/plant_growth_data.csv";

            // (2) Handle the data
            // CSV -> List<List<string>> for all data -> one hot encode -> turn to double[,] -> normalize ->
            // split -> remove and add yTrues to their own double[,] for both train and cross validate

            NetworkFunctions.HandleData(filePath);

            // (3) Build the model

            SupervisedNetwork network = new SupervisedNetwork();

            network.layers = new List<Layer> {
                new Dense(6,6),
                new ReLULayer(),
                new Dense(6,1),
                new SigmoidLayer(),
                new BinaryCrossEntropyLayer()
            };

            // (4) Train the model

            network.Train();

            // (5) Test the model and diagnose

            // (6) Use the model
        }
    }
}
