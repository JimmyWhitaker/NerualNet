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
	private int batchIndex;
	private int[] order; // Order for the data batch
	
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
		
		this.order = new int[numExamples];
		
		for(int i = 0; i < numExamples; i++)
		{
			this.order[i] = i;
		}
		this.batchIndex = 0;
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
		int batchExamples = batchSize;
		int[] rowIndices = new int[batchExamples];
		int[] dataColumnIndices = new int[data.getCols()];
		int[] labelColumnIndices = new int[labels.getCols()];
		
		//Select all data columns
		for(int i = 0; i < dataColumnIndices.length; i++)
		{
			dataColumnIndices[i]=i;
		}
		
		//Select all label columns
		for(int i = 0; i < labelColumnIndices.length; i++)
		{
			labelColumnIndices[i]=i;
		}
		
		//Keep last batch from overflowing
		if( (batchIndex-1) + batchSize > numExamples)
		{
			batchExamples = numExamples - batchIndex;
		}
		
		
		
		// Select rows for the batch
		for(int i = 0; i < rowIndices.length; i++)
		{
			rowIndices[i] = order[batchIndex];
			this.batchIndex++;
		}
		if( batchIndex == numExamples)
		{
			this.batchIndex = 0;
			randomizeOrder();
		}
		
		
		
		// Create Dataset from the selected rows
		Matx batchData = data.select(rowIndices, dataColumnIndices);
		Matx batchLabels = labels.select(rowIndices, labelColumnIndices);
		
		return new MatxDataset(batchData, batchLabels);
	}
	
	/**
	 * Randomize data order using Knuth Shuffle
	 */
	private void randomizeOrder()
	{
		if(numExamples != 1) // Resolve one training example issue
		{
			int randIndex = 0;
			int currentValue = 0;
			for(int i = 0; i < numExamples; i++)
			{
				randIndex = (random.nextInt(numExamples-1));
				currentValue = order[i];
				//swap values
				this.order[i] = order[randIndex];
				this.order[randIndex] = currentValue;
			}
		}
	}
	
	public int getNumBatches()
	{
		return numBatches;
	}
}
