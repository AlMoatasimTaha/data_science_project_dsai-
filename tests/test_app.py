from streamlit.testing.v1 import AppTest
from jobmarket.config import ROOT

def test_streamlit_exploration_and_prediction():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=40).run()
    assert not app.exception
    assert app.title[0].value == "AI Job Market Lab"
    assert len(app.metric) == 3
    if (ROOT / "models/salary/temporal.joblib").exists():
        app.button[0].click().run()
        assert not app.exception
        assert len(app.metric) == 4

def test_empty_filters_do_not_crash():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=40).run()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
