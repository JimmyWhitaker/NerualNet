package com.jmw.image.NeuralNet;

import java.util.Random;

/**
 * Purpose: Manages the data and label matrices
 * 
 * @author Jimmy Whitaker
 *
 */
public class Dataset
{
	private Matx data;
	private Matx labels;
	private int numExamples;
	private static Random random = new Random(1192015);
	
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
	public Dataset(Matx data, Matx labels)
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
		int randRow = 0;
		for(int i = 0; i < this.numExamples; i++)
		{
			randRow = (random.nextInt(this.numExamples-1));
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
}
