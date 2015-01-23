package com.jmw.image.NeuralNet;

import java.util.Random;

/**
 * Purpose: Manages the data and label matrices
 * 
 * @author Jimmy Whitaker
 *
 */
public class MatxDataset implements Dataset
{
	private Matx data;
	private Matx labels;
	protected int numExamples;
	private static Random random = new Random(1192015);
	protected int numBatches;
	protected int batchSize;
	
	/**
	 * Constructs a new, empty Dataset object.
	 */
	public MatxDataset()
	{
		
	}
	
	/**
	 * Constructs a new Dataset object given a data and label Matx.
	 * 
	 * Examples in data should be presented as one per row.
	 * 
	 * An example row in data should correspond to the same row in labels.
	 * 
	 * @param data
	 * @param labels
	 */
	public MatxDataset(Matx data, Matx labels)
	{
		this.data = data;
		this.labels = labels;
		this.numExamples = data.getRows();
	}
	
	/**
	 * Randomizes data examples while keeping correlations 
	 * between data and labels.
	 * 
	 * Uses Knuth Shuffle to randomize rows of data and labels. 
	 */
	public void randomPerm()
	{
		//Resolve one training example issue
		if(numExamples == 1)
		{
			return;
		}
		
		int randRow = 0;
		for(int i = 0; i < numExamples; i++)
		{
			randRow = (random.nextInt(numExamples-1));
			this.data.swapRows(i, randRow);
			this.labels.swapRows(i, randRow);
		}
	}
	
	/**
	 * @return number of examples in the dataset
	 */
	public int getNumExamples()
	{
		return this.numExamples;
	}

	/**
	 * @return the data Matx
	 */
	public Matx getData()
	{
		return this.data;
	}

	/**
	 * @return the label Matx
	 */
	public Matx getLabels()
	{
		return this.labels;
	}
	
	protected void setData(Matx data)
	{
		this.data = data;
	}
	
	protected void setLabels(Matx labels)
	{
		this.labels = labels;
	}

	public void splitIntoBatches(int batchSize)
	{
		this.numBatches = numExamples / batchSize;
		this.batchSize = batchSize;	
	}
	
	public Dataset getBatch()
	{
		return new MatxDataset();
	}
	
	public int getNumBatches()
	{
		return numBatches;
	}
}
