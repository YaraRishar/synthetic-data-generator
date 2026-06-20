"""Веб-интерфейс (Streamlit) для synthetic-data-generator.

Запуск: streamlit run defect_seg/web/app.py
Язык RU/EN переключается в сайдбаре. tensorflow импортируется лениво, только при
запуске обучения.
"""
import glob
import json
import os
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np
import streamlit as st

from defect_seg.web import i18n
from defect_seg.web.i18n import t

DEFAULT_REAL = "example_datasets_scratches/real"
DEFAULT_SYNTH = "example_datasets_scratches/synthetic"


# ---------- вспомогательное ----------------------------------------------

def _dataset_ok(path: str) -> bool:
    return bool(path) and os.path.isdir(os.path.join(path, "images"))


def _image_files(path: str):
    img_dir = os.path.join(path, "images")
    if not os.path.isdir(img_dir):
        return []
    return sorted(f for f in os.listdir(img_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))


def _to_uint8(img):
    return np.clip(img * 255 if img.dtype != np.uint8 else img, 0, 255).astype(np.uint8)


# ---------- вкладки --------------------------------------------------------

def render_help():
    st.header(t("help_header"))
    st.markdown(i18n.help_md())


def render_preview(real, synth):
    from defect_seg import data as seg_data  # noqa: F401
    from defect_seg.cv_io import imread_unicode
    if not _dataset_ok(real):
        st.error(t("err_path_missing", path=real))
        return
    n_real = len(_image_files(real))
    n_synth = len(_image_files(synth)) if _dataset_ok(synth) else 0
    st.success(t("preview_counts", real=n_real, synth=n_synth))

    st.subheader(t("preview_samples"))
    files = _image_files(real)[:3]
    cols = st.columns(max(len(files), 1))
    for col, name in zip(cols, files):
        img = imread_unicode(os.path.join(real, "images", name), cv.IMREAD_GRAYSCALE)
        mask = imread_unicode(os.path.join(real, "bitmaps", name.replace("image", "bitmap")),
                              cv.IMREAD_GRAYSCALE)
        if img is not None:
            col.image(img, caption=name, use_container_width=True)
        if mask is not None:
            col.image(mask, use_container_width=True)


def render_train():
    st.header(t("train_header"))
    c1, c2 = st.columns(2)
    real = c1.text_input(t("real_path_label"), value=DEFAULT_REAL,
                         help=t("real_path_help"), key="real_path")
    synth = c2.text_input(t("synth_path_label"), value=DEFAULT_SYNTH,
                          help=t("synth_path_help"), key="synth_path")

    mode = st.selectbox(t("mode_label"), options=["sweep", "single"],
                        format_func=lambda v: t(f"mode_{v}"), help=t("mode_help"),
                        key="mode")

    g = st.columns(4)
    epochs = g[0].number_input(t("epochs_label"), 1, 100, 3, help=t("epochs_help"), key="epochs")
    batch = g[1].number_input(t("batch_label"), 1, 64, 8, help=t("batch_help"), key="batch")
    test_size = g[2].number_input(t("test_size_label"), 1, 500, 20,
                                  help=t("test_size_help"), key="test_size")
    max_images = g[3].number_input(t("max_images_label"), 0, 100000, 0,
                                   help=t("max_images_help"), key="max_images")

    if mode == "single":
        s = st.columns(2)
        real_size = s[0].slider(t("real_size_label"), 0.0, 1.0, 1.0, 0.1,
                                help=t("real_size_help"), key="real_size")
        synth_size = s[1].slider(t("synth_size_label"), 0.0, 1.0, 0.0, 0.1,
                                 help=t("synth_size_help"), key="synth_size")
    else:
        real_size, synth_size = 1.0, 0.0

    with st.expander(t("advanced_label")):
        threshold = st.slider(t("threshold_label"), 0.0, 1.0, 0.7, 0.05,
                              help=t("threshold_help"), key="threshold")
        lr = st.number_input(t("lr_label"), value=1e-3, format="%.5f",
                             help=t("lr_help"), key="lr")
        seed = st.number_input(t("seed_label"), value=0, step=1,
                               help=t("seed_help"), key="seed")

    b = st.columns(2)
    if b[0].button(t("preview_btn"), key="preview_btn"):
        render_preview(real, synth)

    if b[1].button(t("run_btn"), type="primary", key="run_btn"):
        if not _dataset_ok(real):
            st.error(t("err_path_missing", path=real))
            return
        if not _dataset_ok(synth):
            st.error(t("err_path_missing", path=synth))
            return
        _do_train(real, synth, mode, int(epochs), int(batch), int(test_size),
                  int(max_images), float(real_size), float(synth_size),
                  float(threshold), float(lr), int(seed))


def _do_train(real, synth, mode, epochs, batch, test_size, max_images,
              real_size, synth_size, threshold, lr, seed):
    import tensorflow as tf
    from defect_seg import train
    from defect_seg.config import TrainConfig

    cfg = TrainConfig(real=real, synthetic=synth, epochs=epochs, batch=batch,
                      test_size=test_size, mode=mode, real_size=real_size,
                      synthetic_size=synth_size, threshold=threshold,
                      learning_rate=lr, seed=seed, max_images=max_images)

    bar = st.progress(0.0)
    status = st.empty()
    status.info(t("running"))

    class StProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            bar.progress(min((epoch + 1) / max(epochs, 1), 1.0))

    def on_fraction(idx, total, rs, ss, metrics):
        status.info(t("progress_fraction", idx=idx + 1, total=total, real=rs, synth=ss))
        bar.progress(0.0)

    results = train.run(cfg, keras_callbacks=[StProgress()], on_fraction=on_fraction)
    bar.progress(1.0)
    status.success(t("done_in", sec=results.get("total_time", "?")))

    # таблица + график
    st.subheader(t("results_table"))
    rows = {
        t("chart_synth"): results["synthetic_size"],
        t("chart_iou"): results["iou"],
        t("chart_loss"): results["loss"],
    }
    _render_results_chart(results)
    st.dataframe(rows, use_container_width=True)

    # примеры предсказаний (train.run пишет их в results_dir/predictions)
    overlays = sorted(glob.glob(os.path.join(cfg.results_dir, "predictions", "**", "*_overlay.jpg"),
                                recursive=True), key=os.path.getmtime, reverse=True)[:3]
    if overlays:
        st.subheader(t("samples_title"))
        cols = st.columns(len(overlays))
        from defect_seg.cv_io import imread_unicode
        for col, p in zip(cols, overlays):
            im = imread_unicode(p, cv.IMREAD_COLOR)
            if im is not None:
                col.image(cv.cvtColor(im, cv.COLOR_BGR2RGB), use_container_width=True)


def _render_results_chart(results):
    st.caption(t("chart_title"))
    try:
        import pandas as pd
        df = pd.DataFrame({t("chart_iou"): results["iou"], t("chart_loss"): results["loss"]},
                          index=results["synthetic_size"])
        df.index.name = t("chart_synth")
        st.line_chart(df)
    except Exception:
        st.line_chart({t("chart_iou"): results["iou"], t("chart_loss"): results["loss"]})


def render_contours():
    st.header(t("contours_header"))
    up = st.file_uploader(t("upload_label"), type=["jpg", "jpeg", "png"],
                          help=t("upload_help"), key="bmp")
    what = st.selectbox(t("extract_label"), options=["contours", "bbox", "both"],
                        format_func=lambda v: t(f"extract_{v}"), help=t("extract_help"),
                        key="extract_what")

    if st.button(t("extract_btn"), key="extract_btn") and up is not None:
        from defect_seg import contours as contours_mod
        data = np.frombuffer(up.getvalue(), dtype=np.uint8)
        img = cv.imdecode(data, cv.IMREAD_COLOR)
        with tempfile.TemporaryDirectory() as d:
            tmp = os.path.join(d, "bitmap_дефект.jpg")
            cv.imencode(".jpg", img)[1].tofile(tmp)
            overlay = img.copy()
            if what in ("contours", "both"):
                conts = contours_mod.get_contours(tmp)
                cv.drawContours(overlay, conts, -1, (0, 255, 0), 2)
                st.success(t("contours_found", n=len(conts)))
            if what in ("bbox", "both"):
                boxes = contours_mod.get_bound_box(tmp)
                for x, y, w, h in boxes:
                    cv.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
                st.success(t("bbox_found", n=len(boxes)))

            st.image(cv.cvtColor(overlay, cv.COLOR_BGR2RGB), caption=t("overlay_caption"),
                     use_container_width=True)

            csv_path = os.path.join(d, "contours0_x.csv")
            contours_mod.contours_csv(idx=0, image_path=tmp, path_to_csv=csv_path)
            with open(csv_path, "rb") as f:
                st.download_button(t("download_csv"), f.read(), file_name="contours.csv",
                                   mime="text/csv", key="dl_csv")


def render_results():
    st.header(t("results_header"))
    files = sorted(glob.glob(os.path.join("results", "*.json")))
    if not files:
        st.info(t("results_none"))
        return
    choice = st.selectbox(t("results_source_label"), files, help=t("results_help"),
                          key="results_file")
    with open(choice, encoding="utf-8") as f:
        data = json.load(f)
    st.caption(t("results_summary", n=len(data.get("iou", [])),
                sec=data.get("total_time", "?")))
    _render_results_chart(data)
    st.dataframe({
        t("chart_synth"): data.get("synthetic_size", []),
        t("chart_iou"): data.get("iou", []),
        t("chart_loss"): data.get("loss", []),
    }, use_container_width=True)


def main():
    st.set_page_config(page_title=t("app_title"), layout="wide")
    i18n.language_selector()
    st.title(t("app_title"))
    st.caption(t("app_caption"))

    tab_help, tab_train, tab_contours, tab_results = st.tabs(
        [t("nav_help"), t("nav_train"), t("nav_contours"), t("nav_results")])
    with tab_help:
        render_help()
    with tab_train:
        render_train()
    with tab_contours:
        render_contours()
    with tab_results:
        render_results()


main()
