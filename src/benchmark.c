// What this is:
//	parses CLI arguments (algo and array size)
// 	allocates array - fills with random ints
//	measure RAPL reading before, then after algo execution
// 	measure time
//	runs algo (one from if block)
//	print algo,n,time and energy (J) (converted from uj)
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "utils/rapl.h"
#include "algorithms/insertion.h"
#include "algorithms/merge.h"
#include "algorithms/quick.h"

// Just a convinient variable for unit conversions (ns to s)
#define BILLION 1000000000L

// Fill int Array with random ints in fixed range
void generate_random_array(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000000;
    }
}

int main(int argc, char *argv[]) {
	// Parse and validate CLI args
	// ex- ./bin/benchmark insertion 50000
    if (argc < 3) {
        printf("Usage: %s <algorithm> <size>\n", argv[0]);
        return 1;
    }

    char *algorithm = argv[1];
    int n = atoi(argv[2]);

	// Allocate n ints on the heap, fill with random data
    int *arr = malloc(n * sizeof(int));
    if (!arr) {
        perror("malloc failed");
        return 1;
    }

    generate_random_array(arr, n);

	// Initial time and energy measurements
    struct timespec start, end;

    long long before_uj = rapl_read_uj();

    clock_gettime(CLOCK_MONOTONIC, &start);

	// Algorithm Selection (add more here every time I write new .c .h)
    if (strcmp(algorithm, "insertion") == 0) {
        insertion_sort(arr, (size_t)n);
    } else if (strcmp(algorithm, "merge") == 0) {
        merge_sort(arr, (size_t)n);
    } else if (strcmp(algorithm, "quick") == 0) {
        quick_sort(arr, (size_t)n);
    } else {
        fprintf(stderr, "Unknown algorithm: %s\n", algorithm);
        free(arr);
        return 1;
    }

	// Time and energy measurements after algorithm has completed its task.
    clock_gettime(CLOCK_MONOTONIC, &end);

    long long after_uj = rapl_read_uj();

    double energy_j = rapl_diff_joules(before_uj, after_uj);

    long long elapsed_ns =
        (end.tv_sec - start.tv_sec) * BILLION +
        (end.tv_nsec - start.tv_nsec);

	// Print results, format: isnertion,50000,308221297,2.299554
    printf("%s,%d,%lld,%f\n",
           algorithm,
           n,
           elapsed_ns,
           energy_j);

    free(arr);
    return 0;
}
