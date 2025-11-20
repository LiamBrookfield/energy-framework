#include <stdio.h>
#include <stdlib.h>
#include "rapl.h"

#define RAPL_PATH "/sys/class/powercap/intel-rapl:0/energy_uj"
#define RAPL_MAX_PATH "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"

static long long rapl_max_uj = 0;

static void rapl_init_max() {
    if (rapl_max_uj != 0) return;

    FILE *f = fopen(RAPL_MAX_PATH, "r");
    if (!f) {
        perror("Failed to open max_energy_range_uj");
        exit(1);
    }

    if (fscanf(f, "%lld", &rapl_max_uj) != 1) {
        perror("Failed to read max_energy_range_uj");
        fclose(f);
        exit(1);
    }

    fclose(f);
}

long long rapl_read_uj(void) {
    FILE *f = fopen(RAPL_PATH, "r");
    if (!f) {
        perror("Failed to open energy_uj");
        exit(1);
    }

    long long val;
    if (fscanf(f, "%lld", &val) != 1) {
        perror("Failed to read energy_uj");
        fclose(f);
        exit(1);
    }

    fclose(f);
    return val;
}

double rapl_read_joules(void) {
    long long uj = rapl_read_uj();
    return uj / 1e6;
}

double rapl_diff_joules(long long before_uj, long long after_uj) {
    rapl_init_max();

    long long diff;
    if (after_uj >= before_uj) {
        diff = after_uj - before_uj;
    } else {
        diff = (rapl_max_uj - before_uj) + after_uj;
    }

    return diff / 1e6;
}
