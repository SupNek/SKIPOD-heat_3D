/* Include benchmark-specific header. */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define MIN(x, y) (x < y ? x : y)
#define IDX(i, j, k, n) ((i) * (n) * (n) + (j) * (n) + (k))
#define MAX(x, y) (x > y ? x : y)

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
kernel_heat_3d_MPI(int tsteps, int n, float A[n][n][n],
               float B[n][n][n], int rank, int block_size, int size)
{
    int t, i, j, k;
    int last_block_size = n - block_size * (size-2); // заранее получаем размер последнего блока данных (ширину полосы для последнего процесса)
    float* line = (float *)calloc((block_size+2) * n * n, sizeof(float)); // полоса из массива, передаваемая нулевым процессом
    for (int t = 1; t <= tsteps; t++) {
        if (!rank) {
            for (int num = 1; num < size-1; ++num) {
                // отправляем процессу с номером num соответствующую полосу из А
                MPI_Send(A[MAX((num-1) * block_size - 1, 0)], (block_size + 2) * n * n, MPI_FLOAT, num, 42, MPI_COMM_WORLD);
            }
            // отдельно обрабатываем процесс с номером size-1
            MPI_Send(A[(size-2) * block_size - 1], (last_block_size) * n * n, MPI_FLOAT, size-1, 41, MPI_COMM_WORLD);
        } else if (rank == size - 1) {
            // отдельный прием для size-1
            MPI_Recv(line, (last_block_size) * n * n, MPI_FLOAT, 0, 41, MPI_COMM_WORLD, NULL); // полчаем полосу
        } else {
            // получаем полосу данных из A
            MPI_Recv(line, (block_size+2) * n * n, MPI_FLOAT, 0, 42, MPI_COMM_WORLD, NULL);
        }
        // синхронизируемся, чтобы не было коллизий по отправке сообщений
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank != 0) {
            int block_start = MAX((rank - 1) * block_size, 1); // откуда начинаем цикл по i
            int block_end = MIN(block_start + block_size, n - 1); // конец обрабатываемого блока
            float* block = (float *)calloc((block_end - block_start) * n * n, sizeof(float)); // сюда запишем результат выполнения
            for (i = block_start; i < block_end; i++) {
                for (j = 1; j < n - 1; j++) {
                    for (k = 1; k < n - 1; k++) {
                        block[n * n * (i - block_start) + n * j + k] = 0.25f * line[IDX(i-block_start, j, k, n)];
                        block[n * n * (i - block_start) + n * j + k] += 0.125f * (line[IDX(i-block_start+1, j, k, n)] + line[IDX(i-block_start-1, j, k, n)]
                                                + line[IDX(i-block_start, j+1, k, n)] + line[IDX(i-block_start, j-1, k, n)] +
                                                line[IDX(i-block_start, j, k+1, n)] + line[IDX(i-block_start, j, k-1, n)]);
                    }
                }
            }
            if (rank < size-1) {
                // отправляем полученные данные для обновления B
                MPI_Send(block, (block_end - block_start) * n * n, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
            } else {
                // отдельно обрабатываем процесс size-1
                MPI_Send(block, (block_end - block_start) * n * n, MPI_FLOAT, 0, 4, MPI_COMM_WORLD);
            }
            free(block);
        } else {
            for (int num = 1; num < size-1; ++num) {
                // обновляем В на нулевом процессе
                MPI_Recv(B[block_size*(num-1)], block_size * n * n, MPI_FLOAT, num, 3, MPI_COMM_WORLD, NULL);
            }
            // тут учитываем, что последний процесс может иметь не полный размер
            MPI_Recv(B[block_size*(size-2)], (last_block_size) * n * n, MPI_FLOAT, size-1, 4, MPI_COMM_WORLD, NULL);
        }
        //снова синхронизируемся
        MPI_Barrier(MPI_COMM_WORLD);

        if (!rank) {
            for (int num = 1; num < size-1; ++num) {
                // отправляем процессу с номером num соответствующую полосу из А
                MPI_Send(B[MAX((num-1) * block_size - 1, 0)], (block_size + 2) * n * n, MPI_FLOAT, num, 42, MPI_COMM_WORLD);
            }
            MPI_Send(B[(size-2) * block_size - 1], (last_block_size) * n * n, MPI_FLOAT, size-1, 41, MPI_COMM_WORLD);
        } else if (rank == size - 1) {
            MPI_Recv(line, (last_block_size) * n * n, MPI_FLOAT, 0, 41, MPI_COMM_WORLD, NULL); // получаем полосу
        } else {
            MPI_Recv(line, (block_size+2) * n * n, MPI_FLOAT, 0, 42, MPI_COMM_WORLD, NULL);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank != 0) {
            int block_start = MAX((rank - 1) * block_size, 1);
            int block_end = MIN(block_start + block_size, n - 1);
            float* block = (float *)calloc((block_end - block_start) * n * n, sizeof(float));
            for (i = block_start; i < block_end; i++) {
                for (j = 1; j < n - 1; j++) {
                    for (k = 1; k < n - 1; k++) {
                        block[n * n * (i - block_start) + n * j + k] = 0.25f * line[IDX(i-block_start, j, k, n)];
                        block[n * n * (i - block_start) + n * j + k] += 0.125f * (line[IDX(i-block_start+1, j, k, n)] + line[IDX(i-block_start-1, j, k, n)]
                                                + line[IDX(i-block_start, j+1, k, n)] + line[IDX(i-block_start, j-1, k, n)] +
                                                line[IDX(i-block_start, j, k+1, n)] + line[IDX(i-block_start, j, k-1, n)]);
                    }
                }
            }
            if (rank < size-1) {
                MPI_Send(block, (block_end - block_start) * n * n, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
            } else {
                MPI_Send(block, (block_end - block_start) * n * n, MPI_FLOAT, 0, 4, MPI_COMM_WORLD);
            }
            free(block);
        } else {
            for (int num = 1; num < size-1; ++num) {
                MPI_Recv(A[block_size*(num-1)], block_size * n * n, MPI_FLOAT, num, 3, MPI_COMM_WORLD, NULL);
            }
            // тут учитываем, что последний процесс может иметь не полный размер
            MPI_Recv(A[block_size*(size-2)], (last_block_size) * n * n, MPI_FLOAT, size-1, 4, MPI_COMM_WORLD, NULL);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    free(line);
}

int
main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int n; // размер матрицы
    int tsteps; // число выполняемых операций

    int num_procs; // сюда сохраняем число процессов
    int rank; // по сути id процесса (номер)

    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); // получаем число процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // получаем номер

    if (!rank) {
        printf("\n____Начало выполнения программы____\n");
    }
    printf("Процесс номер %d создан, всего процессов: %d\n", rank, num_procs);

    double parallel[6]; // сюда сохраняем время выполнения

    for (int i = 0; i < 6; i++) {
        parallel[i] = 0;
    }

    for (int k = 1; k <= 5; k++) { // будем усреднять по 5 запускам
        for (int dataset_number = 3; dataset_number <= 5; dataset_number++) {
            switch (dataset_number) {
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
            int block_size = (n + num_procs-2) / (num_procs-1); // получим сколько итераций должен выполнять отдельный процесс
            // num_procs - 1 потому что нулевой процесс не будет участвовать в вычислениях, он будет заниматься пересылкой
            if (!rank) {
                printf("Dataset: %d, ", dataset_number);
            }
            float(*A)[n][n][n];
            float(*B)[n][n][n];
            A = (float(*)[n][n][n]) malloc(n * n * n * sizeof(float));
            B = (float(*)[n][n][n]) malloc(n * n * n * sizeof(float));
            init_array(n, *A, *B);
            double time_start = MPI_Wtime(); // Иницилизация времени старта
            MPI_Barrier(MPI_COMM_WORLD);
            kernel_heat_3d_MPI(tsteps, n, *A, *B, rank, block_size, num_procs);
            MPI_Barrier(MPI_COMM_WORLD);
            double time_finish = MPI_Wtime(); // Иницилизация времени финиша
            double time_execution = time_finish - time_start;
            parallel[dataset_number] += time_execution;
            if (!rank) {
                printf("time in seconds = %0.6lf\n", time_execution); // Вывод времени выполнения программы
            }
            free((void *) A);
            free((void *) B);
        }
    }
    if (!rank) {
        for (int i = 3; i <= 5; i++) {
            printf("Датасет %d, среднее время выполнения %0.6lf\n", i, parallel[i] / 5);
        }
        printf("\n__________________\n");
    }
    MPI_Finalize();
    return 0;
}
