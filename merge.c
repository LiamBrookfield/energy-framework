#include "merge.h"

#include <stddef.h>
#include <stdlib.h>

static void merge_ranges(int *arr, int *aux,
                         size_t left, size_t mid, size_t right) {
    size_t i = left;     // index in left half
    size_t j = mid + 1;  // index in right half
    size_t k = left;     // index in aux

    // Merge two sorted halves into aux
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            aux[k++] = arr[i++];
        } else {
            aux[k++] = arr[j++];
        }
    }

    // Copy any remaining elements from left half
    while (i <= mid) {
        aux[k++] = arr[i++];
    }

    // Copy any remaining elements from right half
    while (j <= right) {
        aux[k++] = arr[j++];
    }

    // Copy merged result back into original array
    for (size_t idx = left; idx <= right; ++idx) {
        arr[idx] = aux[idx];
    }
}

static void merge_sort_recursive(int *arr, int *aux,
                                 size_t left, size_t right) {
    if (left >= right) {
        return; // 0 or 1 element – already sorted
    }

    size_t mid = left + (right - left) / 2;

    merge_sort_recursive(arr, aux, left, mid);
    merge_sort_recursive(arr, aux, mid + 1, right);
    merge_ranges(arr, aux, left, mid, right);
}

void merge_sort(int *arr, size_t n) {
    if (n < 2 || arr == NULL) {
        return;
    }

    // Single aux buffer reused for all merges
    int *aux = (int *)malloc(n * sizeof(int));
    if (!aux) {
        // Allocation failed – fall back to doing nothing
        // (or you could abort; but this keeps the signature simple)
        return;
    }

    merge_sort_recursive(arr, aux, 0, n - 1);

    free(aux);
}
