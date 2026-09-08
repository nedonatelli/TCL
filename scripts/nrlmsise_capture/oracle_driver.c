/* Fixture oracle for the NRL-modified NRLMSISE-00 C implementation.
 *
 * Reads whitespace-separated input records from stdin:
 *   mode doy sec alt_km g_lat g_long lst f107a f107 ap a0 a1 a2 a3 a4 a5 a6
 * mode 0: standard (scalar ap; a0..a6 ignored, pass zeros)
 * mode 1: storm (switches[9]=-1, ap_array from a0..a6)
 * mode 2: like 0 but call gtd7d (effective total density) instead
 * Emits: d[0..8] t[0] t[1] at %.17e, one line per record.
 * All switches set to 1 (SI units), matching NRLMSISE00GasTemp.c.
 */
#include <stdio.h>
#include "nrlmsise-00.h"

int main(void) {
    struct nrlmsise_output out;
    struct nrlmsise_input in;
    struct nrlmsise_flags flags;
    struct ap_array aph;
    int mode, i;
    double ap, a[7];

    while (scanf("%d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                 &mode, &in.doy, &in.sec, &in.alt, &in.g_lat, &in.g_long,
                 &in.lst, &in.f107A, &in.f107, &ap,
                 &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6]) == 17) {
        for (i = 0; i < 24; i++) flags.switches[i] = 1;
        in.year = 0;
        in.ap = ap;
        if (mode == 1) {
            flags.switches[9] = -1;
            for (i = 0; i < 7; i++) aph.a[i] = a[i];
            in.ap_a = &aph;
        } else {
            in.ap_a = 0;
        }
        if (mode == 2) gtd7d(&in, &flags, &out);
        else gtd7(&in, &flags, &out);
        for (i = 0; i < 9; i++) printf("%.17e ", out.d[i]);
        printf("%.17e %.17e\n", out.t[0], out.t[1]);
    }
    return 0;
}
