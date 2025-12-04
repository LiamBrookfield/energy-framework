#include "quick.h"

#include <stddef.h>
#include <stdlib.h>

static void swap(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

// Lomuto partition: uses the last element as pivot
static size_t partition(int *arr, size_t left, size_t right) {
    int pivot = arr[right];
    size_t i = left; // place for next smaller element

    for (size_t j = left; j < right; ++j) {
        if (arr[j] <= pivot) {
            swap(&arr[i], &arr[j]);
            ++i;
        }
    }
    swap(&arr[i], &arr[right]);
    return i;
}

static void quick_sort_recursive(int *arr, size_t left, size_t right) {
    if (left >= right) {
        return;
    }

    size_t p = partition(arr, left, right);

    // Guard against underflow when p == 0
    if (p > 0) {
        quick_sort_recursive(arr, left, p - 1);
    }
    quick_sort_recursive(arr, p + 1, right);
}

void quick_sort(int *arr, size_t n) {
    if (n < 2 || arr == NULL) {
        return;
    }
    quick_sort_recursive(arr, 0, n - 1);
}
