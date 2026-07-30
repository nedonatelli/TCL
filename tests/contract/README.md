# Contract tests

Assertions about the repository rather than the library: that the things shipped
alongside the code actually work.

- `test_examples.py` runs all 30 example scripts and fails on a non-zero exit or
  a printed traceback
- `test_notebook_hygiene.py` checks notebook structure without executing it
- `test_console_encoding.py` checks that printed output survives the Windows
  console codepage
- `test_docs_*.py` check that documented imports resolve, that the architecture
  page's counts match the package, and that its examples run

Every one of these was added after the thing it guards had already broken. The
notebook CI gate ended in `|| echo`, the docs job only failed on ERROR, and
nothing had ever executed the examples.
