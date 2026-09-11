/* CPython extension wrapping the vendored NRLMSISE-00 reference C code.
 *
 * Exposes gtd7/gtd7d (as one entry with an effective-density flag) and
 * ghp7, with all 24 model switches enabled (SI units) and storm mode
 * activated by a 7-element ap sequence -- the same conventions as the
 * MATLAB TCL wrapper and pytcl's pure-Python transcription, which this
 * module supersedes at runtime when importable.
 *
 * The reference implementation keeps state in file-static globals, so
 * a single model call must not be interleaved with another. Every call
 * here runs start-to-finish while holding the GIL, which serializes
 * them; the GIL is deliberately NOT released around the (microsecond
 * scale) computation.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "nrlmsise-00.h"

static int
parse_ap_array(PyObject *ap_obj, struct ap_array *aph, int *storm)
{
    Py_ssize_t i;
    *storm = 0;
    if (ap_obj == Py_None) {
        return 0;
    }
    {
        /* Limited-API (abi3) safe sequence access: PySequence_Fast and
         * its GET_* macros are not in the stable ABI. */
        Py_ssize_t n = PySequence_Size(ap_obj);
        if (n < 0) {
            PyErr_SetString(PyExc_TypeError,
                            "ap_array must be a sequence");
            return -1;
        }
        if (n != 7) {
            PyErr_SetString(PyExc_ValueError,
                            "ap_array must have exactly 7 elements");
            return -1;
        }
        for (i = 0; i < 7; i++) {
            PyObject *item = PySequence_GetItem(ap_obj, i);
            double v;
            if (item == NULL) {
                return -1;
            }
            v = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (v == -1.0 && PyErr_Occurred()) {
                return -1;
            }
            aph->a[i] = v;
        }
    }
    *storm = 1;
    return 0;
}

static void
setup_flags(struct nrlmsise_flags *flags, int storm)
{
    int i;
    for (i = 0; i < 24; i++) {
        flags->switches[i] = 1;
    }
    if (storm) {
        flags->switches[9] = -1;
    }
}

static PyObject *
build_output(const struct nrlmsise_output *out)
{
    PyObject *d = PyTuple_New(9);
    PyObject *t = PyTuple_New(2);
    PyObject *result;
    int i;
    if (d == NULL || t == NULL) {
        Py_XDECREF(d);
        Py_XDECREF(t);
        return NULL;
    }
    /* PyTuple_SetItem (the checked, ref-stealing function) is in the
     * limited API; the SET_ITEM macro is not. */
    for (i = 0; i < 9; i++) {
        PyTuple_SetItem(d, i, PyFloat_FromDouble(out->d[i]));
    }
    for (i = 0; i < 2; i++) {
        PyTuple_SetItem(t, i, PyFloat_FromDouble(out->t[i]));
    }
    result = PyTuple_Pack(2, d, t);
    Py_DECREF(d);
    Py_DECREF(t);
    return result;
}

static PyObject *
py_gtd7(PyObject *self, PyObject *args)
{
    int doy;
    double sec, alt, g_lat, g_long, lst, f107a, f107, ap;
    PyObject *ap_obj;
    int effective;
    struct nrlmsise_input input;
    struct nrlmsise_flags flags;
    struct nrlmsise_output output;
    struct ap_array aph;
    int storm;

    if (!PyArg_ParseTuple(args, "iddddddddOp", &doy, &sec, &alt, &g_lat,
                          &g_long, &lst, &f107a, &f107, &ap, &ap_obj,
                          &effective)) {
        return NULL;
    }
    if (parse_ap_array(ap_obj, &aph, &storm) < 0) {
        return NULL;
    }
    setup_flags(&flags, storm);
    input.year = 0;
    input.doy = doy;
    input.sec = sec;
    input.alt = alt;
    input.g_lat = g_lat;
    input.g_long = g_long;
    input.lst = lst;
    input.f107A = f107a;
    input.f107 = f107;
    input.ap = ap;
    input.ap_a = storm ? &aph : NULL;

    if (effective) {
        gtd7d(&input, &flags, &output);
    }
    else {
        gtd7(&input, &flags, &output);
    }
    return build_output(&output);
}

static PyObject *
py_ghp7(PyObject *self, PyObject *args)
{
    int doy;
    double sec, press, g_lat, g_long, lst, f107a, f107, ap;
    PyObject *ap_obj;
    struct nrlmsise_input input;
    struct nrlmsise_flags flags;
    struct nrlmsise_output output;
    struct ap_array aph;
    int storm;
    double alt;
    PyObject *out_obj, *result;

    if (!PyArg_ParseTuple(args, "iddddddddO", &doy, &sec, &press, &g_lat,
                          &g_long, &lst, &f107a, &f107, &ap, &ap_obj)) {
        return NULL;
    }
    if (parse_ap_array(ap_obj, &aph, &storm) < 0) {
        return NULL;
    }
    setup_flags(&flags, storm);
    input.year = 0;
    input.doy = doy;
    input.sec = sec;
    input.alt = 0.0;
    input.g_lat = g_lat;
    input.g_long = g_long;
    input.lst = lst;
    input.f107A = f107a;
    input.f107 = f107;
    input.ap = ap;
    input.ap_a = storm ? &aph : NULL;

    alt = ghp7(&input, &flags, &output, press);
    out_obj = build_output(&output);
    if (out_obj == NULL) {
        return NULL;
    }
    result = Py_BuildValue("dO", alt, out_obj);
    Py_DECREF(out_obj);
    return result;
}

static PyMethodDef Nrlmsise00Methods[] = {
    {"gtd7", py_gtd7, METH_VARARGS,
     "gtd7(doy, sec, alt_km, g_lat, g_long, lst, f107a, f107, ap, "
     "ap_array_or_None, effective) -> ((d0..d8), (t0, t1))"},
    {"ghp7", py_ghp7, METH_VARARGS,
     "ghp7(doy, sec, press_hpa, g_lat, g_long, lst, f107a, f107, ap, "
     "ap_array_or_None) -> (alt_km, ((d0..d8), (t0, t1)))"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef nrlmsise00module = {
    PyModuleDef_HEAD_INIT,
    "_nrlmsise00_c",
    "Compiled NRLMSISE-00 reference implementation.",
    -1,
    Nrlmsise00Methods,
};

PyMODINIT_FUNC
PyInit__nrlmsise00_c(void)
{
    return PyModule_Create(&nrlmsise00module);
}
