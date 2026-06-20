"""Тесты веб-интерфейса (Streamlit AppTest) для synthetic-data-generator.

Запуск без pytest:  python tests/test_streamlit_app.py
Запуск с pytest:    pytest tests/test_streamlit_app.py
"""
import glob
import os
import shutil
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(APP_DIR, "streamlit_app.py")


def _run():
    at = AppTest.from_file(APP)
    at.run(timeout=90)
    return at


def test_app_runs_no_exception():
    at = _run()
    assert not at.exception


def test_default_language_is_ru():
    at = _run()
    assert "Верификатор" in at.title[0].value


def test_switch_to_english():
    at = _run()
    at.selectbox(key="lang").set_value("en").run(timeout=90)
    assert "Synthetic Data Verifier" in at.title[0].value
    assert at.selectbox(key="mode").label == "Mode"


def test_mode_combobox_options_localized():
    at = _run()
    assert at.selectbox(key="mode").options == [
        "Перебор доли синтетики (sweep)", "Один прогон (single)"]
    at.selectbox(key="lang").set_value("en").run(timeout=90)
    assert at.selectbox(key="mode").options == [
        "Sweep synthetic fraction (sweep)", "Single run (single)"]


def test_extract_combobox_options_localized():
    at = _run()
    assert at.selectbox(key="extract_what").options == [
        "Контуры", "Ограничивающие прямоугольники", "Контуры и прямоугольники"]
    at.selectbox(key="lang").set_value("en").run(timeout=90)
    assert at.selectbox(key="extract_what").options == [
        "Contours", "Bounding boxes", "Contours and boxes"]


def test_combobox_value_is_canonical_regardless_of_language():
    # бэкенд должен получать каноническое значение, а не локализованную подпись
    at = _run()
    assert at.selectbox(key="mode").value == "sweep"
    at.selectbox(key="lang").set_value("en").run(timeout=90)
    assert at.selectbox(key="mode").value == "sweep"


def test_widget_help_is_localized():
    at = _run()
    ru = next(ni for ni in at.number_input if ni.key == "epochs")
    assert "проход" in (ru.help or "")
    at.selectbox(key="lang").set_value("en").run(timeout=90)
    en = next(ni for ni in at.number_input if ni.key == "epochs")
    assert "passes" in (en.help or "")


def test_help_text_localized():
    at = _run()
    md = " ".join(m.value for m in at.markdown)
    assert "О программе" in md
    at.selectbox(key="lang").set_value("en").run(timeout=90)
    md = " ".join(m.value for m in at.markdown)
    assert "About" in md


def test_invalid_path_shows_error():
    at = _run()
    next(ti for ti in at.text_input if ti.key == "real_path").set_value("nope/missing").run(timeout=90)
    at.button(key="run_btn").click().run(timeout=90)
    assert not at.exception
    assert any("nope/missing" in e.value for e in at.error)


def test_tiny_training_through_ui():
    if not os.path.isdir("example_datasets_scratches/real/images"):
        print("  (пропуск: нет example датасета)")
        return
    at = _run()
    at.selectbox(key="mode").set_value("single").run(timeout=90)
    for key, val in [("max_images", 8), ("test_size", 2), ("epochs", 1), ("batch", 2)]:
        next(ni for ni in at.number_input if ni.key == key).set_value(val).run(timeout=90)
    next(s for s in at.slider if s.key == "real_size").set_value(0.5).run(timeout=90)
    next(s for s in at.slider if s.key == "synth_size").set_value(0.5).run(timeout=90)
    try:
        at.button(key="run_btn").click().run(timeout=600)
        assert not at.exception, at.exception
        assert len(at.dataframe) >= 1            # таблица метрик отрисована
    finally:
        shutil.rmtree("results", ignore_errors=True)
        for d in glob.glob("example_datasets_scratches/real/predictions/pred*"):
            shutil.rmtree(d, ignore_errors=True)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} прошли")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
