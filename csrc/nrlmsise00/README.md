# NRLMSISE-00 reference C implementation (vendored)

Dominik Brodowski's public-domain C port of the NRLMSISE-00 empirical
atmosphere model, as modified and vendored by the U.S. Naval Research
Laboratory's Tracker Component Library
(`3rd_Party_Libraries/nrlmsise-00-bc9a2fe`, MATLAB tree). The NRL
modifications (documented in the MATLAB tree's "Tracker Library
Changes.txt") are: `ghp7` returns the computed altitude, MEX memory
and printing hooks, and shadowing-warning renames. `mex.h` here is
pytcl's shim so the code compiles standalone: the memory hooks map
back to `malloc`/`free`, and `mexPrintf` routes to `nrlmsise_report`
in `pymodule.c`, which queues the message and raises it as a Python
`UserWarning` once the model call returns (instead of printing to
native stdout).

Provenance: MATLAB TCL commit a9acd8f. License: public domain -- see
License.txt (Brodowski's release statement, contingent on the original
Fortran being US-government public domain, which it is).

Compiled into `pytcl.atmosphere._nrlmsise00_c` by `setup.py` /
`csrc/nrlmsise00/pymodule.c`. The pure-Python transcription in
`pytcl/atmosphere/nrlmsise00.py` is the fallback when the extension is
unavailable and the cross-validation target for it.
