package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import shared.filt.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class IrisANN {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 4, hiddenLayer = 5, outputLayer = 1, itr = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {

        TestTrainSplitFilter tt = new TestTrainSplitFilter(80);
        tt.filter(set);
        DataSet trainSet = tt.getTrainingSet();
        DataSet testSet = tt.getTestingSet();

        // backprop for control
        // BackPropagationNetwork network = new BackPropagationNetwork();
        // network = factory.createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});
       
        // double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        // for(int i = 0; i < itr; i++) {

        //     double error = 0;
        //     Instance[] trainInstances = trainSet.getInstances(); 
        //     for(int j = 0; j < trainInstances.length; j++) {
        //         network.setInputValues(trainInstances[j].getData());
        //         network.run();
        //         network.backpropagate();
        //         network.updateWeights(new StandardUpdateRule());

        //         Instance output = trainInstances[j].getLabel(), example = new Instance(network.getOutputValues());
        //         example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
        //         error += measure.value(output, example);
        //     }

        //     System.out.println(String.format("%04d,%s", i, df.format(error)));
        // }
        // end = System.nanoTime();
        // trainingTime = end - start;
        // trainingTime /= Math.pow(10,9);

        // end backprop

        // for (int itr = 1; itr < 501; itr++) {

            for(int i = 0; i < oa.length; i++) {
                networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
                nnop[i] = new NeuralNetworkOptimizationProblem(trainSet, networks[i], measure);
            }

            oa[0] = new RandomizedHillClimbing(nnop[0]);
            oa[1] = new SimulatedAnnealing(1E11, .55, nnop[1]);
            oa[2] = new StandardGeneticAlgorithm(100, 100, 50, nnop[2]);

            for(int i = 0; i < oa.length; i++) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                // start = System.nanoTime();
                // correct = 0;
                // incorrect = 0;
                train(oa[i], networks[i], oaNames[i], itr, trainSet); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10,9);

                Instance optimalInstance = oa[i].getOptimal();
                networks[i].setWeights(optimalInstance.getData());

                double predicted, actual;
                start = System.nanoTime();
                Instance[] testInputs = testSet.getInstances();
                for(int j = 0; j < testInputs.length; j++) {
                    networks[i].setInputValues(testInputs[j].getData());
                    networks[i].run();

                    actual = Double.parseDouble(testInputs[j].getLabel().toString());
                    predicted = Double.parseDouble(networks[i].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10,9);

                System.out.printf("%d,%s\n", itr, df.format(correct/(correct+incorrect)*100));
                
                results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                            "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                            + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                            + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
                // break; 
            }
        // }
        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int trainingIterations, DataSet trainSet) {
        // System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            Instance[] trainInstances = trainSet.getInstances(); 
            for(int j = 0; j < trainInstances.length; j++) {
                network.setInputValues(trainInstances[j].getData());
                network.run();

                Instance output = trainInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            // System.out.println(String.format("%04d,%s", i, df.format(error)));
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[100][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/iris2.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[4]; // 4 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 4; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());
                
                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1]));
        }

        return instances;
    }
}
