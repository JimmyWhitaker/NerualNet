package com.jmw.image.NeuralNet;

import java.io.IOException;
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
	private int matxIndex;
	
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
		this.data = new Matx(data);
		this.labels = new Matx(labels);
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
		return numExamples;
	}

	/**
	 * @return the data Matx
	 */
	public Matx getData()
	{
		return data;
	}

	/**
	 * @return the label Matx
	 */
	public Matx getLabels()
	{
		return labels;
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
	
	/**
	 * TODO return a batch (currently returns all)
	 */
	public Dataset getBatch()
	{
		int numExamples = batchSize;
		
		//Keep last batch from overflowing
		if( (matxIndex + batchSize) > data.getRows()-1)
		{
			numExamples = data.getRows() - matxIndex;
		}
			
		return new MatxDataset(data,labels);
	}
	
	public int getNumBatches()
	{
		return numBatches;
	}
}
