/* Minimal shim so the NRL MEX-ified NRLMSISE-00 compiles standalone. */
#ifndef MEX_SHIM_H
#define MEX_SHIM_H
#include <stdlib.h>
#include <stdio.h>
#define mxMalloc  malloc
#define mxFree    free
#define mexPrintf printf
#endif
