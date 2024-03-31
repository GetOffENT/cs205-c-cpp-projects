import java.util.InputMismatchException;
import java.util.Scanner;
import java.util.Random;

public class MatrixMultiplication {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        Random random = new Random();

        int mode;
        int rows1, cols1, rows2, cols2;
        while (true) {
            try {
                System.out.print("Enter 1 for user input mode, 2 for random generation mode: ");
                mode = scanner.nextInt();
                if (mode == 1 || mode == 2) {
                    break; // Break the loop if the input is valid
                } else {
                    System.out.println("Invalid input, please enter 1 or 2.");
                }
            } catch (InputMismatchException e) {
                System.out.println("Invalid input, please enter an integer.");
                scanner.nextLine(); // Clear the input buffer
            }
        }

        while (true) {
            try {
                System.out.print("Enter rows and columns for matrix 1: ");
                rows1 = scanner.nextInt();
                cols1 = scanner.nextInt();

                System.out.print("Enter rows and columns for matrix 2: ");
                rows2 = scanner.nextInt();
                cols2 = scanner.nextInt();

                if (cols1 != rows2) {
                    System.out.println("Matrix multiplication is not possible, the number of columns in matrix 1 must be equal to the number of rows in matrix 2.");
                } else {
                    break; // Break the loop if the input is valid
                }
            } catch (InputMismatchException e) {
                System.out.println("Invalid input, please enter integers.");
                scanner.nextLine(); // Clear the input buffer
            }
        }

        float[][] matrix1 = new float[rows1][cols1];
        float[][] matrix2 = new float[rows2][cols2];

        if (mode == 1) {
            // User input mode
            readMatrixFromUser(matrix1, scanner);
            readMatrixFromUser(matrix2, scanner);
        } else if (mode == 2) {
            // Random generation mode
            fillMatrixWithRandomValues(matrix1, random);
            System.out.println("Matrix 1:");
            printMatrix(matrix1);
            fillMatrixWithRandomValues(matrix2, random);
            System.out.println("Matrix 2:");
            printMatrix(matrix2);
        } else {
            System.out.println("Invalid mode selected.");
            return;
        }

        long startTime = System.currentTimeMillis();
        float[][] result = multiplyMatrices(matrix1, matrix2);
        long endTime = System.currentTimeMillis();

        double duration = endTime - startTime; // milliseconds

        if (result != null) {
            System.out.println("Resultant Matrix:");
            printMatrix(result);
            System.out.println("Time taken for matrix multiplication: " + duration + " milliseconds");
        }
    }

    private static void fillMatrixWithRandomValues(float[][] matrix, Random random) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = random.nextFloat() * 100;
            }
        }
    }

    private static void readMatrixFromUser(float[][] matrix, Scanner scanner) {
        System.out.printf("Enter values for a %dx%d matrix:\n", matrix.length, matrix[0].length);
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                while (true) { // Infinite loop to keep trying until a valid input is received
                    try {
                        System.out.printf("Enter value for element [%d][%d]: ", i, j);
                        matrix[i][j] = scanner.nextFloat();
                        break; // Break the loop if the input is valid
                    } catch (InputMismatchException e) {
                        System.out.println("Invalid input, please enter a valid floating point number.");
                        scanner.nextLine(); // Clear the input buffer
                    }
                }
            }
        }
    }

    private static float[][] multiplyMatrices(float[][] mat1, float[][] mat2) {
        float[][] result = new float[mat1.length][mat2[0].length];
        for (int i = 0; i < mat1.length; i++) {
            for (int j = 0; j < mat2[0].length; j++) {
                float sum = 0;
                for (int k = 0; k < mat1[0].length; k++) {
                    sum += mat1[i][k] * mat2[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    private static void printMatrix(float[][] matrix) {
        for (float[] row : matrix) {
            for (float element : row) {
                System.out.printf("%7.2f ", element);
            }
            System.out.println();
        }
    }
}
