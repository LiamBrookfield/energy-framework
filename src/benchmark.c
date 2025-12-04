#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "utils/rapl.h"
#include "algorithms/insertion.h"
#include "algorithms/merge.h"
#include "algorithms/quick.h"

#define BILLION 1000000000L

void generate_random_array(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000000;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <algorithm> <size>\n", argv[0]);
        return 1;
    }

    char *algorithm = argv[1];
    int n = atoi(argv[2]);

    int *arr = malloc(n * sizeof(int));
    if (!arr) {
        perror("malloc failed");
        return 1;
    }

    generate_random_array(arr, n);

    struct timespec start, end;

    long long before_uj = rapl_read_uj();

    clock_gettime(CLOCK_MONOTONIC, &start);

    if (strcmp(algorithm, "insertion") == 0) {
        insertion_sort(arr, n);
    } else {
        printf("Unknown algorithm: %s\n", algorithm);
        free(arr);
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    long long after_uj = rapl_read_uj();

    double energy_j = rapl_diff_joules(before_uj, after_uj);

    long long elapsed_ns =
        (end.tv_sec - start.tv_sec) * BILLION +
        (end.tv_nsec - start.tv_nsec);

    printf("%s,%d,%lld,%f\n",
           algorithm,
           n,
           elapsed_ns,
           energy_j);

    free(arr);
    return 0;
}
