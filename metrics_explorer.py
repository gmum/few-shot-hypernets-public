# run with:
# streamlit run metrics_explorer.py
from typing import Tuple, Dict, List, Union, Any

import numpy as np
import streamlit as st
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import base64
from collections import defaultdict

st.set_page_config(
    page_title="FSL metrics",
    page_icon=None,
    layout="wide",
)

"""# FSLH - metrics explorer"""

root = Path("/home/mprzewie/coding/gmum_mnt/uj/few-shot-hypernets/save/checkpoints/cross_char/")

METRICS_FILE = "metrics.json"
ARGS_FILE = "args.json"

LOADING_PROGRESS = st.progress(0.0)

PROGRESS_TEXT = st.empty()
HN_PREFIX = "hn"


def metrics_dict_to_df(
        experiment_name: str,
        metrics_dict: Dict[str, List[Union[float, List[float]]]],
        args_dict: Dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for m_name, values in metrics_dict.items():
        for e, vls in enumerate(values):
            vls = [vls] if not isinstance(vls, list) else [np.mean(vls)] #vls
            rows.extend([{
                "exp_name": experiment_name,
                "met_name": m_name,
                "epoch": e,
                "value": v,
                **{ak: av for (ak, av) in args_dict.items() if ak.startswith(HN_PREFIX)}
            } for v in vls])
    return pd.DataFrame(rows)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_experiment(experiment_path: Path) -> Tuple[Dict, pd.DataFrame]:
    with (experiment_path / METRICS_FILE).open("r") as f:
        metrics = json.load(f)
    with (experiment_path / ARGS_FILE).open("r") as f:
        args = json.load(f)
    return args, metrics_dict_to_df(experiment_name=experiment_path.name, metrics_dict=metrics, args_dict=args)


loggable_experiments = {p.parent.name: p for p in root.glob(f"*/{METRICS_FILE}") if (p.parent / ARGS_FILE).exists()}

experiment_args = dict()
experiment_metrics = dict()

loggable_experiments = list(loggable_experiments.items())  # [:3]

for i, (e, p) in enumerate(loggable_experiments):
    PROGRESS_TEXT.text(f"loading {p.parent.name} {i}/{len(loggable_experiments)}")

    try:
        args, metrics_df = load_experiment(p.parent)
        experiment_args[e] = args
        experiment_metrics[e] = metrics_df

        LOADING_PROGRESS.progress((i + 1) / len(loggable_experiments))
    except Exception as exc:
        print(exc)
        pass

PROGRESS_TEXT.text(f"Loaded {len(experiment_metrics)} experiments")

df = pd.concat([mdf for mdf in experiment_metrics.values()])

available_metrics = sorted(df.met_name.unique())
all_args = sorted({a for ad in experiment_args.values() for a in ad.keys()})


"""## Selected metric over the course of epochs"""

selected_metric = st.selectbox("Select metric", available_metrics, index=available_metrics.index("accuracy_val_max"))
aggregate_y = st.checkbox("Aggregate Y?", value=True)

st.altair_chart(alt.Chart(
    df[df.met_name == selected_metric],
).mark_line(point=True).encode(
    x="epoch",
    y=alt.Y("value", aggregate=("mean" if aggregate_y else alt.Undefined)),
    color="exp_name", tooltip=["exp_name", "value", "epoch"] + [a for a in all_args if a.startswith(HN_PREFIX)],
).configure_legend(labelLimit=0).interactive().properties(title=selected_metric), use_container_width=True)


"""## How do hyperparams influence the metric?"""

for a in all_args:
    if a.startswith(HN_PREFIX):
        unique_as = sorted(df[a].unique())
        with st.expander(f"{a} := {unique_as}", expanded=(len(unique_as) > 1)):
            st.altair_chart(alt.Chart(
                df[df.met_name == selected_metric],
            ).mark_line(point=True).encode(
                x="epoch",
                y=alt.Y("value", aggregate=("mean" if aggregate_y else alt.Undefined)),
                color="exp_name", tooltip=["exp_name", "value", "epoch"] + [a for a in all_args if a.startswith(HN_PREFIX)],
                column=a
            ).configure_legend(labelLimit=0).interactive().properties(title=selected_metric))