package com.jmw.image.NeuralNet;

/**
 * Softmax Activation Function
 * 
 * @author Jimmy
 */
public class Softmax extends ActivationFunction
{
	/**
	 * Determines if a de-serialized file is compatible with this class.
	 */
	private static final long serialVersionUID = 4706188892893788298L;

	/**
	 * Constructs new Sigmoid Activation Function
	 */
	public Softmax()
	{
		
	}
	
	/**
	 * Computes the element-wise output of the activation function.
	 * 
	 * @param matx input Matx
	 * @return computed output Matx
	 */
	@Override
	public Matx getOutput(Matx matx) 
	{
		Matx result = new Matx(matx); 
		
		//Iterate through each column of output data
		for(int j = 0; j < matx.getCols(); j++)
		{
			double current = 0;
			double max = 0; 
			double numerator = 0;
			double denominator = 0;
			
			/* 
			 * Find the max of the column.
			 * By multiplying the numerator and denominator by
			 * the max value in column we prevent overflow.
			 */
			max = result.maxInColumn(j);
			
			//Compute denominator and store numerator
			for(int i = 0; i < matx.getRows(); i++)
			{
				current = result.get(i, j);
				numerator = Math.exp(current - max);
				result.set(i, j, numerator); // Don't have to calculate e^x again.
				denominator += numerator;
			}
			
			//Compute output of softmax function
			for(int i = 0; i < matx.getRows(); i++)
			{
				double ans = (result.get(i, j))/denominator;
				result.set(i,j,ans);
			}
		}
		return result;                
	}
	
	@Override
	protected double getOutput(double net)
	{
		throw new UnsupportedOperationException("Not possible without a single element.");
	}

	/**
	 * Computes an element-wise derivative output of the activation function.
	 * 
	 * @param matx input Matx
	 * @return computed output Matx
	 */
	@Override
	public Matx getDerivative(Matx matx)
	{
		Matx result = this.getOutput(matx);
		
		//Iterate through each column
		for(int j = 0; j < matx.getCols(); j++)
		{
			double current = 0;
			double answer = 0;
			
			//Compute derivative of value for the softmax function and store it.
			for(int i = 0; i < matx.getRows(); i++)
			{
				current = result.get(i, j);
				answer = current * (1d - current);
				result.set(i, j, answer);
			}
			
		}
		return result;
	}
	
	@Override
	protected double getDerivative(double net) 
	{
		throw new UnsupportedOperationException("Not possible without a single element.");
	}

	@Override
	public String getType()
	{
		return "Softmax";
	}

}

