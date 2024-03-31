import java.util.InputMismatchException;
import java.util.Random;
import java.util.Scanner;

public class MatrixMultiplicationTest {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        Random random = new Random();

        int rowsAndCols;
        int rows1, cols1, rows2, cols2;

        rowsAndCols = scanner.nextInt();
        rows1 = cols1 = rows2 = cols2 = rowsAndCols;

        int T = 100;
        double time_sum = 0;

        for (int t = 0; t < T; t++) {
            float[][] matrix1 = new float[rows1][cols1];
            float[][] matrix2 = new float[rows2][cols2];

            fillMatrixWithRandomValues(matrix1, random);
//        System.out.println("Matrix 1:");
//        printMatrix(matrix1);
            fillMatrixWithRandomValues(matrix2, random);
//        System.out.println("Matrix 2:");
//        printMatrix(matrix2);

            long startTime = System.nanoTime();
            float[][] result = multiplyMatrices(matrix1, matrix2);
            long endTime = System.nanoTime();

            double duration = (double) (endTime - startTime) / 1000000; // milliseconds
            time_sum += duration;

//        System.out.println("Resultant Matrix:");
//        printMatrix(result);
//        System.out.println("Time taken for matrix multiplication: " + duration + " milliseconds");
//            System.out.println(duration);
        }
//        System.out.printf("Average time taken for matrix multiplication: %.6f  milliseconds", time_sum / T);
    }

    private static void fillMatrixWithRandomValues(float[][] matrix, Random random) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = random.nextFloat() * 100;
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
