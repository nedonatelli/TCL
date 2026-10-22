/* Minimal shim so the NRL MEX-ified NRLMSISE-00 compiles standalone. */
#ifndef MEX_SHIM_H
#define MEX_SHIM_H
#include <stdlib.h>
#include <stdio.h>
#define mxMalloc  malloc
#define mxFree    free
/* Failure reports are queued (pymodule.c) and issued as one Python
 * RuntimeWarning per model call; a bare printf bypassed the warning
 * machinery entirely. */
void nrlmsise_report(const char *fmt, ...);
#define mexPrintf nrlmsise_report
#endif
