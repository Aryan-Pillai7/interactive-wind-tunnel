from streamlit.testing.v1 import AppTest


def test_app_renders_and_runs_solver():
    at = AppTest.from_file("app/app.py", default_timeout=180).run()
    assert not at.exception
    assert any("Compute time" in m.label for m in at.metric)  # solver mode by default
    at.radio[0].set_value("Neural surrogate").run()
    assert not at.exception
    assert any("Inference time" in m.label for m in at.metric)
    at.button[0].click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Velocity rel. L2 error" in labels or at.error
