"""Feature-local isolation for the shared external-boundary recorders."""

import pytest


@pytest.fixture(autouse=True)
def reset_feature_externals(web_external):
    web_external.reset()
    yield
    if web_external.search_gate is not None:
        web_external.search_gate.set()


@pytest.fixture
def error_text():
    def extract(result) -> str:
        return "\n".join(getattr(item, "text", "") for item in (result.content or []))

    return extract
