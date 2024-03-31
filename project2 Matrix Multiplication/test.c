#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to allocate memory for a matrix stored as a single array
float *createMatrix(int rows, int cols)
{
    return (float *)malloc(rows * cols * sizeof(float));
}

// Function to free memory of a matrix
void freeMatrix(float *matrix)
{
    free(matrix);
}

// Function to fill a matrix with random values
void fillMatrixWithRandomValues(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = (float)rand() / RAND_MAX * 1000; // Random values between 0 and 100
    }
}

// Function to multiply two matrices
float *multiplyMatrices(float *mat1, int rows1, int cols1, float *mat2, int rows2, int cols2)
{
    float *result = createMatrix(rows1, cols2);

    for (int i = 0; i < rows1; i++)
    {
        for (int j = 0; j < cols2; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < cols1; k++)
            {
                sum += mat1[i * cols1 + k] * mat2[k * cols2 + j];
            }
            result[i * cols2 + j] = sum;
        }
    }

    return result;
}

// Function to print a matrix
void printMatrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%7.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main()
{
    srand(time(NULL)); // Seed for random number generation
    int rowsAndCols;
    int rows1, cols1, rows2, cols2;
    int returnValue = 0;

    printf("Enter rows and columns for matrix 1 and matrix 2: ");
    returnValue = scanf("%d", &rowsAndCols);
    rows1 = cols1 = rows2 = cols2 = rowsAndCols;

    int T = 100;
    double time_sum = 0;

    for (int i = 0; i < T; i++)
    {
        float *matrix1 = createMatrix(rows1, cols1);
        float *matrix2 = createMatrix(rows2, cols2);
        fillMatrixWithRandomValues(matrix1, rows1, cols1);
        fillMatrixWithRandomValues(matrix2, rows2, cols2);
        // printf("Matrix 2:\n");
        // printMatrix(matrix2, rows2, cols2);

        clock_t start, end;
        double cpu_time_used;

        start = clock();
        // Execute matrix multiplication
        float *result = multiplyMatrices(matrix1, rows1, cols1, matrix2, rows2, cols2);
        end = clock();

        cpu_time_used = ((double)(end - start)) / 1000;
        time_sum += cpu_time_used;

        // printf("Resultant Matrix:\n");
        // printMatrix(result, rows1, cols2);
        // printf("Time taken for matrix multiplication: %f milliseconds\n", cpu_time_used);
        printf("%f\n", cpu_time_used);
        freeMatrix(result);

        freeMatrix(matrix1);
        freeMatrix(matrix2);
    }

    printf("Average time taken for matrix multiplication: %f milliseconds\n", time_sum / T);

    return 0;
}