package com.jmw.image.NeuralNet;

import java.io.IOException;
import java.util.Random;

import mnist.MnistManager.MnistManager;

/**
 * Utilizes the MnistManager provided by the mnist-tools project.
 *  
 * 
 * @author Jimmy
 *
 */
public class MnistDataset implements Dataset
{
	protected int numExamples;
	protected int numBatches;
	protected int batchSize;
	private MnistManager mnistManager; 
	private int mnistIndex; //Index of the mnistManager
	private final int IMAGE_LENGTH = 784; //Number of pixels in an mnist image
	private final int LABEL_LENGTH = 10; //Vector that holds a 1 in the index of the value (Used for softmax)
	
	public MnistDataset(String imageFile, String labelFile)
	{
		try
		{
			this.mnistManager = new MnistManager(imageFile, labelFile);
			this.mnistIndex = 1;
			this.mnistManager.setCurrent(mnistIndex);
			
			if(mnistManager.getImages().getCount() == mnistManager.getLabels().getCount())
			{
				this.numExamples = mnistManager.getImages().getCount();
			}else{
				throw new Exception("Image and Label files have different lengths.");
			}
		} catch (Exception e)
		{
			e.printStackTrace();
		}
		
	}

	/**
	 * Create and return a Dataset with a given number of examples.
	 * Examples are returned in order and mini-batches can be created.
	 * 
	 * @param numExamples
	 * @return
	 */
	public Dataset getBatch()
	{
		int numExamples = batchSize;
		
		//Keep last batch from overflowing
		if( (mnistIndex-1) + batchSize > mnistManager.getImages().getCount())
		{
			numExamples = mnistManager.getImages().getCount() - mnistIndex;
		}
			
		double[] image = null;
		double[][] data = new double[numExamples][IMAGE_LENGTH];
		double[] label = null;
		double[][] labels = new double[numExamples][LABEL_LENGTH];


		for(int i = 0; i < numExamples; i++)
		{
			//Get example from MnistManager
			try
			{
				/* Warning: methods readBytes from file, so 
				 * anything that calls these methods will iterate
				 * the dataset.
				 */
				image = mnistManager.readProcessedImage();
				label = mnistManager.readProcessedLabel();

			} catch (IOException e){
				e.printStackTrace();
			}

			//Add example to data array
			for(int j = 0; j<image.length; j++)
			{
				data[i][j] = image[j];
			}

			//Add label to label array
			for(int j = 0; j<label.length; j++)
			{
				labels[i][j] = label[j];
			}

			//Increment mnistManager reference
			this.mnistIndex++;
			if( mnistIndex > mnistManager.getImages().getCount())
				this.mnistIndex = 1;
			this.mnistManager.setCurrent(mnistIndex);
		}

		return new MatxDataset(Matx.createMatx(data), Matx.createMatx(labels));
	}

	public int getNumFeatures()
	{
		return IMAGE_LENGTH;
	}

	public void randomPerm()
	{
		throw new UnsupportedOperationException("Operation Performed in MatxDataset.");		
	}

	public int getNumExamples()
	{
		return numExamples;
	}

	/**
	 * Returnd Matx of all Data.
	 */
	public Matx getData()
	{	
		int startingIndex = mnistIndex;
		double[] image = null;
		double[][] data = new double[numExamples][IMAGE_LENGTH];

		for(int i = 0; i < numExamples; i++)
		{
			//Get example from MnistManager
			try
			{
				/* Warning: methods readBytes from file, so 
				 * anything that calls these methods will iterate
				 * the dataset.
				 */
				image = mnistManager.readProcessedImage();

			} catch (IOException e){
				e.printStackTrace();
			}

			//Add example to data array
			for(int j = 0; j<image.length; j++)
			{
				data[i][j] = image[j];
			}

			//Increment mnistManager reference
			this.mnistManager.setCurrent(i+1);
		}
		this.mnistManager.setCurrent(startingIndex);
		
		return Matx.createMatx(data);
	}

	public Matx getLabels()
	{
		int startingIndex = mnistIndex;
		double[] label = null;
		double[][] labels = new double[numExamples][LABEL_LENGTH];
		
		for(int i = 0; i < numExamples; i++)
		{
			//Get example from MnistManager
			try
			{
				/* Warning: methods readBytes from file, so 
				 * anything that calls these methods will iterate
				 * the dataset.
				 */
				label = mnistManager.readProcessedLabel();

			} catch (IOException e){
				e.printStackTrace();
			}

			//Add label to label array
			for(int j = 0; j<label.length; j++)
			{
				labels[i][j] = label[j];
			}

			//Increment mnistManager reference
			this.mnistManager.setCurrent(i+1);
		}
		this.mnistManager.setCurrent(startingIndex);

		return Matx.createMatx(labels);
	}

	public void splitIntoBatches(int batchSize)
	{
		this.numBatches = numExamples / batchSize;
		this.batchSize = batchSize;	
	}
	
	public int getNumBatches()
	{
		return numBatches;
	}
}
