using Encog.Engine.Network.Activation;
using Encog.MathUtil.Randomize;
using Encog.ML.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Neural.Networks.Training.Propagation.Quick;
using Encog.Neural.Networks.Training.Propagation.SCG;
using Encog.Neural.Networks.Training.Propagation.Manhattan;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ENCOG3
{
    class Program
    {
        private static BasicNetwork CreateNetwork()
        {
            BasicNetwork basickNetwork = new BasicNetwork();
            basickNetwork.AddLayer(new BasicLayer(null, true, 2));
            basickNetwork.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 4));
            basickNetwork.AddLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
            basickNetwork.Structure.FinalizeStructure();
            basickNetwork.Reset();

            return basickNetwork;
        }
        static void Main(string[] args)
        {
            double[][] XOR_input =
            {
                new [] {0.0, 0.0},
                new [] {1.0, 0.0},
                new [] {0.0, 1.0},
                new [] {1.0, 1.0}
            };

            double[][] XOR_ideal =
            {
                new [] {0.0},
                new [] {1.0},
                new [] {1.0},
                new [] {0.0}
            };

            var trainingSet = new BasicMLDataSet(XOR_input, XOR_ideal);

            var network = CreateNetwork();

            // understand difference between different propagations
            var resilientTraining = new ResilientPropagation(network, trainingSet);
            var backpropagationTraining = new Backpropagation(network, trainingSet);
            var quickTraining = new QuickPropagation(network, trainingSet);
            var manhattanTraining = new ManhattanPropagation(network, trainingSet, 0.1); // experiment with diff vals
            var SCGTraining = new ScaledConjugateGradient(network, trainingSet);

            using (System.IO.StreamWriter file = new System.IO.StreamWriter(@"./data.txt"))
            {
                foreach (var item in trainingSet)
                {

                    var output = network.Compute(item.Input);
                    file.WriteLine("Input: {0}, {1}; Ideal: {2}; Actual: {3};",
                        item.Input[0], item.Input[1], item.Ideal[0], output[0]);
                }

                int iteration = 0;
                do
                {
                    backpropagationTraining.Iteration();
                    iteration++;
                    file.WriteLine("Iteration {0}, Error {1}", iteration, backpropagationTraining.Error);
                } while (backpropagationTraining.Error > 0.001 && iteration < 10000);

                foreach (var item in trainingSet)
                {

                    var output = network.Compute(item.Input);
                    file.WriteLine("Input: {0}, {1}; Ideal: {2}; Actual: {3};",
                        item.Input[0], item.Input[1], item.Ideal[0], output[0]);
                }
            }
        }
    }
}
















