package com.jmw.image.NeuralNet;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;

import mnist.MnistManager.MnistManager;

/**
 * Utilizes the MnistManager provided by the mnist-tools project.
 *  
 * 
 * @author Jimmy
 *
 */
public class MnistDataset implements Dataset, Serializable
{
	/**
	 * Determines if a de-serialized file is compatible with this class.
	 */
	private static final long serialVersionUID = -1417962337337680L;
	
	private final int IMAGE_LENGTH = 784; //Number of pixels in an mnist image
	private final int LABEL_LENGTH = 10; //Vector that holds a 1 in the index of the value (Used for softmax)
	
	private Matx data;
	private Matx labels;
	
	private int[] order; // Order for the data batch
	private int mnistIndex; //Index of the mnistManager
	private static Random random = new Random(1192015); //TODO add parameter for seed
	
	protected int numExamples;
	protected int numBatches;
	protected int batchSize;
	
	private transient MnistManager mnistManager; 
	
	/**
	 * Constructs a new MnistDataset given an image and label file
	 * 
	 * @param imageFile
	 * @param labelFile
	 */
	public MnistDataset(String imageFile, String labelFile)
	{
		try
		{
			this.mnistManager = new MnistManager(imageFile, labelFile);
			
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
		
		this.data = importData();
		this.labels = importLabels();
		this.order = new int[numExamples];
		
		for(int i = 0; i < numExamples; i++)
		{
			this.order[i] = i;
		}
		this.mnistIndex = 0;
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
		int batchExamples = batchSize;
		//Keep last batch from overflowing
		if( (mnistIndex-1) + batchSize > numExamples)
		{
			batchExamples = numExamples - mnistIndex;
		}
		
		int[] rowIndices = new int[batchExamples];
		int[] dataColumnIndices = new int[IMAGE_LENGTH];
		int[] labelColumnIndices = new int[LABEL_LENGTH];
		
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
		
		// Select rows for the batch
		for(int i = 0; i < rowIndices.length; i++)
		{
			rowIndices[i] = order[mnistIndex];
			this.mnistIndex++;
		}
		if( mnistIndex == numExamples)
		{
			this.mnistIndex = 0;
			randomizeOrder();
		}
		
		
		
		// Create Dataset from the selected rows
		Matx batchData = data.select(rowIndices, dataColumnIndices);
		Matx batchLabels = labels.select(rowIndices, labelColumnIndices);
		
		return new MatxDataset(batchData, batchLabels);
	}

	/**
	 * @return number of image features
	 */
	public int getNumImageFeatures()
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
	 * @return Matx of all data examples
	 */
	private Matx importData()
	{	
		this.mnistManager.setCurrent(1);
		
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
		
		return Matx.createMatx(data);
	}

	/**
	 * @return Matx of all label examples
	 */
	private Matx importLabels()
	{
		this.mnistManager.setCurrent(1);
		
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

		return Matx.createMatx(labels);
	}
	
	/**
	 * Split the data into batches and randomize the order of the examples.
	 * 
	 * @param batch size
	 */
	public void splitIntoBatches(int batchSize)
	{
		this.numBatches = numExamples / batchSize;
		this.batchSize = batchSize;	

		//Randomize order data using Knuth Shuffle
		randomizeOrder();
	}
	
	/**
	 * Randomize data order using Knuth Shuffle
	 */
	public void randomizeOrder()
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
	
	/**
	 * @return the number of Batches
	 */
	public int getNumBatches()
	{
		return numBatches;
	}
	
	/**
	 * @return Matx copy of the data
	 */
	public Matx getData()
	{
		return data.copy();
	}
	
	/**
	 * @return Matx copy of the labels
	 */
	public Matx getLabels()
	{
		return labels.copy();
	}
	
	/**
	 * Save a serialized version of the MnistDataset
	 * 
	 * @param filename name (location) of ouput file typically ending with (.ser)
	 */
	public void save(String filename)
	{
		try
		{
			FileOutputStream fileOut = new FileOutputStream(filename);
			ObjectOutputStream outStream = new ObjectOutputStream(fileOut);
			outStream.writeObject(this);
			outStream.close();
			fileOut.close();
		}catch(IOException i)
		{
			i.printStackTrace();
		}
	}
	
	/**
	 * De-serialize MnistDataset object.
	 * @param filename location of file
	 * @return de-serialized MnistDataset 
	 */
	public static MnistDataset load(String filename)
	{
		//Deserialize
		System.out.println("************");
		System.out.println("Deserializing");
		MnistDataset mnistDataset = null;
		try
		{
			FileInputStream fileIn =new FileInputStream(filename);
			ObjectInputStream in = new ObjectInputStream(fileIn);
			mnistDataset = (MnistDataset) in.readObject();
			in.close();
			fileIn.close();
		}catch(IOException i)
		{
			i.printStackTrace();
			return null;
		}catch(ClassNotFoundException c)
		{
			System.out.println("NeuralNet class not found.");
			c.printStackTrace();
			return null;
		}
		System.out.println("Done Deserializing MNIST Dataset " + filename);
		return mnistDataset;   
	}
}
