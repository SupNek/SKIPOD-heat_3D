/* Include benchmark-specific header. */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define MIN(x, y) (x < y ? x : y)

static void
init_array(int n, float A[n][n][n],
           float B[n][n][n]) // инициализируем массивы A и B
{
    int i, j, k;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                A[i][j][k] = B[i][j][k] = (float) (i + j + (n - k)) * 10 / (n);
            }
        }
    }
}

static void
kernel_heat_3d(int tsteps, int n, float A[n][n][n],
               float B[n][n][n]) // основная функция работы с массивами
{
    int t, i, j, k;

    for (t = 1; t <= tsteps; t++) {
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
                for (k = 1; k < n - 1; k++) {
                    B[i][j][k] = 0.25f * A[i][j][k];
                    B[i][j][k] += 0.125f * (A[i + 1][j][k] + A[i - 1][j][k] + A[i][j + 1][k] + A[i][j - 1][k] +
                                            A[i][j][k + 1] + A[i][j][k - 1]);
                }
            }
        }
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
                for (k = 1; k < n - 1; k++) {
                    A[i][j][k] = 0.25f * B[i][j][k];
                    A[i][j][k] += 0.125f * (B[i + 1][j][k] + B[i - 1][j][k] + B[i][j + 1][k] + B[i][j - 1][k] +
                                            B[i][j][k + 1] + B[i][j][k - 1]);
                }
            }
        }
    }
}

static void
kernel_heat_3d_parallel_task(int tsteps, int n, float A[n][n][n],
                             float B[n][n][n]) // реализация через механизм задач
{

    int t, i, j, k;
#pragma omp parallel
    {
#pragma omp single
        {
            for (t = 1; t <= tsteps; t++) {
                for (i = 1; i < n - 1; i++) {
#pragma omp task shared(A, B, n) firstprivate(t, i) private(j, k)
                    for (j = 1; j < n - 1; j++) {
                        for (k = 1; k < n - 1; k++) {
                            B[i][j][k] = 0.25f * A[i][j][k];
                            B[i][j][k] += 0.125f * (A[i + 1][j][k] + A[i - 1][j][k] + A[i][j + 1][k] + A[i][j - 1][k] +
                                                    A[i][j][k + 1] + A[i][j][k - 1]);
                        }
                    }
                }
#pragma omp taskwait

                for (i = 1; i < n - 1; i++) {
#pragma omp task shared(A, B, n) firstprivate(t, i) private(j, k)
                    for (j = 1; j < n - 1; j++) {
                        for (k = 1; k < n - 1; k++) {
                            A[i][j][k] = 0.25f * B[i][j][k];
                            A[i][j][k] += 0.125f * (B[i + 1][j][k] + B[i - 1][j][k] + B[i][j + 1][k] + B[i][j - 1][k] +
                                                    B[i][j][k + 1] + B[i][j][k - 1]);
                        }
                    }
                }
#pragma omp taskwait
            }
        }
    }
}

int
main(int argc, char **argv)
{
    int threads_num = omp_get_max_threads();
    printf("\n____Начало выполнения программы____\n");
    printf("Число используемых нитей: %d\n", threads_num); // получаем число нитей

    int n;
    int tsteps;
    // вместо заголовочного файла реализуем выбор размера датасета с консоли:
    double parallel[6]; // сюда сохраняем время выполнения

    for (int i = 0; i < 6; i++) {
        parallel[i] = 0;
    }

    for (int k = 1; k <= 5; k++) { // будем усреднять по 5 запускам
        for (int dataset_number = 1; dataset_number <= 5; dataset_number++) {
            switch (dataset_number) {
            case 1:
                n = 10;
                tsteps = 20;
                break;
            case 2:
                n = 20;
                tsteps = 40;
                break;
            case 3:
                n = 40;
                tsteps = 100;
                break;
            case 4:
                n = 120;
                tsteps = 500;
                break;
            case 5:
                n = 200;
                tsteps = 1000;
                break;
            default:
                break;
            }
            printf("Dataset: %d, ", dataset_number);
            float(*A)[n][n][n];
            float(*B)[n][n][n];

            A = (float(*)[n][n][n]) malloc(n * n * n * sizeof(float));
            B = (float(*)[n][n][n]) malloc(n * n * n * sizeof(float));
            init_array(n, *A, *B);
            double time_start = omp_get_wtime(); // Иницилизация времени старта
            kernel_heat_3d_parallel_task(tsteps, n, *A, *B);
            // kernel_heat_3d(tsteps, n, *A, *B);
            double time_finish = omp_get_wtime(); // Иницилизация времени финиша
            double time_execution = time_finish - time_start;
            parallel[dataset_number] += time_execution;
            printf("time in seconds = %0.6lf\n", time_execution); // Вывод времени выполнения программы
            free((void *) A);
            free((void *) B);
        }
    }

    for (int i = 1; i <= 5; i++) {
        printf("Датасет %d, среднее время выполнения %0.6lf\n", i, parallel[i] / 5);
    }
    printf("\n__________________\n");

    return 0;
}
