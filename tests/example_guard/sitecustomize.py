"""Turn a stray ``fig.show()`` into a hard failure.

``tests/contract/test_examples.py`` puts this directory on PYTHONPATH when it runs the
example scripts; Python imports ``sitecustomize`` automatically at startup.

Without it, an unguarded ``show()`` in a headless CI container does not raise
-- plotly falls back to writing a temp file and returns -- so the example
would still exit 0 while quietly doing the wrong thing.
"""


def _blocked(self, *args, **kwargs):
    raise AssertionError(
        "fig.show() was called while PYTCL_SHOW_PLOTS=0. Wrap it as:\n"
        "    if SHOW_PLOTS:\n"
        "        fig.show()\n"
        "    else:\n"
        '        fig.write_html(str(OUTPUT_DIR / "name.html"))'
    )


try:
    import plotly.basedatatypes

    plotly.basedatatypes.BaseFigure.show = _blocked
except ImportError:  # plotly is an optional extra
    pass
