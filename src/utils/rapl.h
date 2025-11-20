#ifndef RAPL_H
#define RAPL_H

long long rapl_read_uj(void);
double rapl_read_joules(void);
double rapl_diff_joules(long long before_uj, long long after_uj);

#endif
