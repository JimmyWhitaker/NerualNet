package com.jmw.image.NeuralNet;

import java.util.Random;

import org.la4j.matrix.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;

/**
 * Generalization of a Matrix Implementation to accommodate different libraries. 
 * 
 * @author Jimmy Whitaker
 */
public class Matx
{
	Basic2DMatrix matrix;
	private static Random random = new Random(1192015); //TODO parameter for seed
	// TODO make mean and st dev parameters
	private static final double MEAN = 0;
	private static final double STDDEV = 0.5;
	
	
	/**
	 * Constructor for a new Matx object which contains a Basic2DMatrix 
	 * from the la4j library.
	 */
	public Matx()
	{
		this.matrix = new Basic2DMatrix();
	}
	
	/**
	 * Creates a new Matx object given a la4j matrix.
	 * @param matrix
	 */
	private Matx(Matrix matrix)
	{
		this.matrix = (Basic2DMatrix)matrix.copy();
	}
	
	/**
	 * Creates a new Matx object given another Matx object.
	 * This constructor creates a copy of the current Matx
	 * object.
	 * @param matx
	 */
	public Matx(Matx matx) {
		this.matrix = (Basic2DMatrix) matx.matrix.copy();
	}

	/**
	 * Multiply two matrices together and return their result.
	 * 
	 * @param matx1 
	 * @param matx2
	 * @return new matrix that is the product of the parameters
	 */
	public static Matx multiply(Matx matx1, Matx matx2)
	{
		Matrix result = matx1.matrix.multiply(matx2.matrix);
		return new Matx(result);
	}
	
	/**
	 * Create Matx object from a 2D double array.
	 * 
	 * @param matx
	 * @return new Matx object 
	 */
	public static Matx createMatx(double[][] matx)
	{
		Matrix result = org.la4j.matrix.dense.Basic2DMatrix.from2DArray(matx);
		return new Matx(result);
	}
	
	/**
	 * Create Matx object from a 1D double array.
	 * 
	 * @param matx
	 * @return new Matx object 
	 */
	public static Matx createMatx(double[] matx)
	{
		Matrix result = org.la4j.matrix.dense.Basic2DMatrix.from1DArray(matx.length, 1, matx);
		return new Matx(result);
	}
	
	/**
	 * Create a Matx object with a matrix rows x cols that contains
	 * random values taken from a normal distribution with the class 
	 * parameters MEAN and STDDEV.
	 */
	public static Matx createRandNormMatx(int rows, int cols)
	{		
		Matrix result =  Basic2DMatrix.zero(rows, cols);
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				result.set(i, j, (random.nextGaussian()*STDDEV) + MEAN);
			}
		}
		return new Matx(result);
	}
	
	/**
	 * Subtract matrices.
	 * 
	 * @param matx1
	 * @param matx2
	 * @return new Matx object
	 */
	public static Matx subtract(Matx matx1, Matx matx2)
	{
		Matrix result = matx1.matrix.subtract(matx2.matrix);
		return new Matx(result);
	}
	
	/**
	 * Add matrices.
	 * 
	 * @param matx1
	 * @param matx2
	 * @return new Matx object
	 */
	public static Matx add(Matx matx1, Matx matx2)
	{
		Matrix result = matx1.matrix.add(matx2.matrix);
		return new Matx(result);
	}
	
	/**
	 * Element-wise multiplication (Hadamard product)
	 * 
	 * @param matx1
	 * @param matx2
	 * @return new Matx object
	 */
	public static Matx elementMultiply(Matx matx1, Matx matx2)
	{
		Matrix result = matx1.matrix.hadamardProduct(matx2.matrix);
		return new Matx(result);
	}

	/**
	 * Compute transpose of a Matx object
	 * 
	 * @return new Matx object
	 */
	public Matx getTranspose()
	{
		return new Matx(this.matrix.transpose());
	}
	
	/**
	 * Multiply Matx object by a scalar
	 * 
	 * @param scalar
	 * @param matx
	 * @return new Matx object
	 */
	public static Matx scalarMultiply(double scalar, Matx matx)
	{
		Matrix result = matx.matrix.multiply(scalar);
		return new Matx(result);
	}
	
	/**
	 * Get value at the given row and col in the matrix.
	 * 
	 * @param row
	 * @param col
	 * @return double value
	 */
	public double get(int row, int col)
	{
		return this.matrix.get(row, col);
	}
	
	/**
	 * Set a value in the matrix at row x col.
	 * 
	 * @param row
	 * @param col
	 * @param value
	 */
	public void set(int row, int col, double value)
	{
		this.matrix.set(row, col, value);
	}
	
	/**
	 * @return number of columns in the matrix
	 */
	public int getCols()
	{
		return this.matrix.columns();
	}

	/**
	 * @return number of rows in the matrix
	 */
	public int getRows()
	{
		return this.matrix.rows();
	}
	
	/**
	 * Convert Matx object to a String
	 */
	@Override
	public String toString()
	{
		return this.matrix.toString();
	}

	/**
	 * Return the ith row of a matrix.
	 * 
	 * @param i
	 * @return new Matx object
	 */
	public Matx getRow(int i)
	{
		return new Matx(this.matrix.getRow(i).toRowMatrix());
	}
	
	/**
	 * Creates a copy of a Matx object
	 * 
	 * @return new Matx object
	 */
	public Matx copy()
	{
		return new Matx(this.matrix.copy());
	}
	
	/**
	 * Swaps rows i and j in a matrix.
	 * @param i
	 * @param j
	 */
	public void swapRows(int i, int j)
	{
		this.matrix.swapRows(i, j);
	}
	
}
