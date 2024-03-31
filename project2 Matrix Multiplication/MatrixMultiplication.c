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
        matrix[i] = (float)rand() / RAND_MAX * 100; // Random values between 0 and 100
    }
}

// Function to read matrix values from user
void readMatrixFromUser(float *matrix, int rows, int cols)
{
    printf("Enter values for a %dx%d matrix:\n", rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            while (1) // Infinite loop to keep trying until a valid input is received
            {
                printf("Enter value for element [%d][%d]: ", i, j);
                if (scanf("%f", &matrix[i * cols + j]) == 1)
                {
                    break; // Break the loop if the input is valid
                }
                else
                {
                    printf("Invalid input, please enter a valid floating point number.\n");
                    while (getchar() != '\n')
                        ; // Clear the input buffer
                }
            }
        }
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
    int rows1, cols1, rows2, cols2, mode;
    int returnValue = 0;

    printf("Enter 1 for user input mode, 2 for random generation mode: ");
    while (1) // Infinite loop to keep trying until a valid input is received
    {
        returnValue = scanf("%d", &mode);

        if (returnValue == 1 && (mode == 1 || mode == 2))
        {
            break; // Break the loop if the input is valid
        }
        else
        {
            printf("Invalid input, please enter 1 or 2.\n");
            while (getchar() != '\n')
                ; // Clear the input buffer to remove invalid input
        }
    }

    while (1) // Infinite loop to keep trying until a valid input is received
    {
        printf("Enter rows and columns for matrix 1: ");
        returnValue = scanf("%d %d", &rows1, &cols1);
        if (returnValue != 2)
        { // Check if two integers were received
            printf("Invalid input, please enter two integers.\n");
            while (getchar() != '\n')
                ;     // Clear the input buffer
            continue; // Skip the rest of the loop and retry
        }

        printf("Enter rows and columns for matrix 2: ");
        returnValue = scanf("%d %d", &rows2, &cols2);
        if (returnValue != 2)
        { // Check if two integers were received
            printf("Invalid input, please enter two integers.\n");
            while (getchar() != '\n')
                ;     // Clear the input buffer
            continue; // Skip the rest of the loop and retry
        }

        if (cols1 != rows2)
        {
            printf("Matrix multiplication is not possible, the number of columns in matrix 1 must be equal to the number of rows in matrix 2.\n");
            continue; // Retry the whole process again
        }

        break; // Valid input received, exit the loop
    }

    float *matrix1 = createMatrix(rows1, cols1);
    float *matrix2 = createMatrix(rows2, cols2);

    if (mode == 1)
    {
        // User input mode
        readMatrixFromUser(matrix1, rows1, cols1);
        readMatrixFromUser(matrix2, rows2, cols2);
    }
    else if (mode == 2)
    {
        // Random generation mode
        fillMatrixWithRandomValues(matrix1, rows1, cols1);
        printf("Matrix 1:\n");
        printMatrix(matrix1, rows1, cols1);
        fillMatrixWithRandomValues(matrix2, rows2, cols2);
        printf("Matrix 2:\n");
        printMatrix(matrix2, rows2, cols2);
    }
    else
    {
        printf("Invalid mode selected.\n");
        return 1;
    }

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    // Execute matrix multiplication
    float *result = multiplyMatrices(matrix1, rows1, cols1, matrix2, rows2, cols2);
    end = clock();

    cpu_time_used = ((double)(end - start)) / 1000;

    printf("Resultant Matrix:\n");
    printMatrix(result, rows1, cols2);
    printf("Time taken for matrix multiplication: %f milliseconds\n", cpu_time_used);
    freeMatrix(result);

    freeMatrix(matrix1);
    freeMatrix(matrix2);

    return 0;
}