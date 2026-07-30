# API surface tests

The shape of the public interface rather than the correctness of a computation:
what a package exports, what a signature accepts, which exception type comes
back and what it carries.

These fail when a rename, a moved symbol, or a changed error type breaks a
caller who was relying on the published surface -- the kind of break that unit
tests pass straight through because the arithmetic still works.
