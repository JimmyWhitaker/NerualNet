package com.jmw.image.NeuralNet;

/**
 * Purpose: Manages the data and label matrices
 * 
 * @author Jimmy Whitaker
 *
 */
public interface Dataset
{	
	/**
	 * Randomizes data examples while keeping correlations 
	 * between data and labels.
	 * 
	 * Uses Knuth Shuffle to randomize rows of data and labels. 
	 */
	public void randomPerm();
	
	/**
	 * @return number of examples in the dataset
	 */
	public int getNumExamples();

	/**
	 * @return the data Matx
	 */
	public Matx getData();

	/**
	 * @return the label Matx
	 */
	public Matx getLabels();

	/*
	 * Determine number of examples in batch size and number of Batches
	 */
	public void splitIntoBatches(int batchSize);
	
	public Dataset getBatch();

	public int getNumBatches();
}
