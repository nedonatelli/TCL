"""The public surface of the placeholder packages.

pytcl.misc, pytcl.physical_values, pytcl.scheduling and pytcl.transponders are
declared but carry no modules. This pins that: they must export nothing, so an
accidental export is visible rather than silently becoming public API.
"""


class TestEmptyNamespaceModules:
    def test_stub_modules_are_empty(self):
        import pytcl.misc
        import pytcl.physical_values
        import pytcl.scheduling
        import pytcl.transponders

        for mod in (
            pytcl.physical_values,
            pytcl.scheduling,
            pytcl.transponders,
            pytcl.misc,
        ):
            assert mod.__all__ == []
