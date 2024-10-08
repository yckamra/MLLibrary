﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using ML;

namespace MLDriver
{
    class MainClass
    {
        // TODO: fix batch so
        // that near end of epoch if current batch is smaller than the batch size we still update weights and biases,
        // if above initializations do not improve model too much add Dropout and L2 Regularization...
        // we can also try adding Adam optimization which includes momentum
        
        public static void Main(string[] args)
        {
            // (1) Copy your CSV to CSV folder in driver and do:
            // Right-click the CSV -> Properties -> Copy to output directory -> always copy
            
            // (2) Create a filePath
            string CSVFileName = "plant_growth_data.csv"; // <--- change this to your CSV including .csv

            string relativePath = "CSV/" + CSVFileName; // no touch
            string pathToCSVDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CSV/"); // no touch
            string fullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, relativePath); // no touch

            // (3) Handle the data
            // CSV -> List<List<string>> for all features and List<List<string>> for targets -> one hot encode features ->
            // one hot encode targets if needed -> turn to double[,] for both features and targets -> normalize features ->
            // shuffle? -> split

            double[,] featureTrainData = null; // no touch
            double[,] featureTestData = null; // no touch
            double[,] targetTrainData = null; // no touch
            double[,] targetTestData = null; // no touch
            UI.HandleData(fullPath, pathToCSVDirectory, ref featureTrainData, ref featureTestData, ref targetTrainData, ref targetTestData); // mo touch

            // (4) Build the model

            SupervisedNetwork network = new SupervisedNetwork(); // no touch

            network.layers = new List<Layer> { // <--- Add and subtract layers as you please (below is example network)
                new Dense(9,8, "HeInitialization"),
                new ReLULayer(),
                new Dense(8, 8, "HeInitialization"),
                new ReLULayer(),
                new Dense(8, 1, "XavierInitialization"),
                new SigmoidLayer(),
                new BinaryCrossEntropyLayer()
            };

            // (5) Train the model

            UI.HandleSupervisedTrain(network, featureTrainData, targetTrainData, featureTestData, targetTestData);

            // (6) Use the model // TODO: remember that any new inputs that we put in MUST BE NORMALIZED

        }
    }
}
