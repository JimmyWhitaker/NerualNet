package com.jmw.image.NeuralNet;

/**
 * NeuralNet Implementation
 * App.java
 * Purpose: Main class to run a Neural Network Tests.
 * 
 * @author Jimmy Whitaker
 * @version 0.1 1/21/15
 */
public class App 
{
	/**
	 * The main method for the NeuralNet program.
	 *
	 * @param args Not used
	 */
    public static void main( String[] args )
    {
    	//Compute Runtime
    	long start = System.currentTimeMillis();
    	
    	//XOR Test
    	int[] neuronsPerLayer = {2,1};
    	String[] layerType = {"Tanh","Tanh"}; // Currently all neurons in a layer have the same activation function
    	int epochs = 15;
    	double learningRate = 0.2;
    	double momentum = 1;
    	
    	double[][] xor_data = {{0,0,1,1},{0,1,0,1}};
    	double[] xor_labels = {0,1,1,0};
    	
    	Dataset dataset = new Dataset(Matx.createMatx(xor_data).getTranspose(), Matx.createMatx(xor_labels));
    	//TODO add mean and stddev to parameters
    	NeuralNet nn = new NeuralNet(neuronsPerLayer, layerType, dataset.getData().getCols()); 
    	
    	nn.train(dataset, epochs, learningRate, momentum);
    	nn.test(dataset,true);
    	
    	//Error Discovery Test - Comparison by Hand
//    	double[] data = {1,0,1};
//    	double[] label = {1};
//    	Matx dataset = Matx.createMatx(data).getTranspose();
//    	Matx dataLabels = Matx.createMatx(label);
//    	NeuralNet nn = new NeuralNet(neuronsPerLayer, dataset, dataLabels);
//    	double[][] w1 = {{0.2,0.4,-0.5},{-0.3,0.1,0.2}};
//    	double[] w2 = {-0.3,-0.2};
//    	nn.layers[0].weight = Matx.createMatx(w1);
//    	nn.layers[1].weight = Matx.createMatx(w2).getTranspose();
//    	double[][] b1 = {{-0.4},{0.2}};
//    	double[][] b2 = {{0.1}};
//    	nn.layers[0].bias = Matx.createMatx(b1);
//    	nn.layers[1].bias = Matx.createMatx(b2);
//    	nn.train(dataset, dataLabels, 1, 0.9, 1);
//    	nn.test(dataset, dataLabels);
    	System.out.println("Runtime: " + (System.currentTimeMillis()-start));
    	System.exit(0);
    }
}
