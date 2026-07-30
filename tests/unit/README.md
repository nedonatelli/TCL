# Unit tests

One function or class at a time, against values worked out independently of the
implementation.

A test belongs here if it exercises a single unit and the expected value comes
from the mathematics rather than from running the code and recording what it
said. If the expected value came from an outside implementation, it belongs in
`validation/` instead.

These are the default. When in doubt, a test starts here and moves out only if
one of the narrower directories clearly fits.
