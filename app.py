import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from local_components import card_container
from datetime import datetime

st.set_page_config(
    page_title="Only Finance",
    layout="wide"
)

st.markdown("""
    <style>
        div.block-container {padding-top:1rem;}
        div.block-container {padding-bottom:1rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>Only Finance</h1>", unsafe_allow_html=True)

def process_data(df):
    transition_idx = df.columns.get_loc("P U T S")
    first_level = ["CALLS"] * transition_idx + ["PUTS"] * (len(df.columns) - transition_idx)
    second_level = df.iloc[0]

    multi_index = pd.MultiIndex.from_tuples(zip(first_level, second_level))
    df.columns = multi_index

    df = df.drop(0)
    df[("CALLS", "Pos")] = df[("CALLS", "Pos")].ffill()
    df = df.dropna(how="all", subset=df.columns.difference([("CALLS", "Pos")]))
    df[("CALLS", "Pos")] = df[("CALLS", "Pos")].apply(lambda x: re.search(r"\d{2} \w{3} \d{2}", x).group(0) if re.search(r"\d{2} \w{3} \d{2}", x) else None)
    df[("CALLS", "Pos")] = df[("CALLS", "Pos")].apply(lambda x: datetime.strptime(x, "%d %b %y").date())

    for col in df.columns[1:]:
        df[col] = df[col].str.replace(",", "")
        df[col] = df[col].str.rstrip("%")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)
    df = df.reset_index(drop=True)
    df.columns = ["_".join(col).strip() for col in df.columns.values]

    return df

def filter_consecutive_zero_groups_at_ends(df, epsilon=1e-1):
    row_sums = df.iloc[:, 2:].abs().sum(axis=1)

    non_zero_indices = row_sums[row_sums > epsilon].index
    if non_zero_indices.empty:
        return df

    first_non_zero_idx = non_zero_indices.min()
    last_non_zero_idx = non_zero_indices.max()

    filtered_df = df.loc[first_non_zero_idx:last_non_zero_idx].reset_index(drop=True)

    return filtered_df

def plot_exposure(filtered_df, exposure_type, filter_zero_values):
    st.header(exposure_type.title())
    
    exposure_columns = [f"Call_{exposure_type.title()}_Exposure", f"Put_{exposure_type.title()}_Exposure", f"Net_{exposure_type.title()}_Exposure"]
    exposure_filtered_df = filtered_df[["Row_Number", "CALLS_Strike"] + exposure_columns]

    if filter_zero_values:
        exposure_filtered_df = filter_consecutive_zero_groups_at_ends(exposure_filtered_df)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=exposure_filtered_df["CALLS_Strike"],
        y=exposure_filtered_df[exposure_columns[0]],
        name=f"Call {exposure_type.title()} Exposure",
        marker_color="rgb(141, 211, 199)",
        hovertemplate="<b>Strike Price</b>: %{x}<br><b>Value</b>: %{y}",
    ))

    fig.add_trace(go.Bar(
        x=exposure_filtered_df["CALLS_Strike"],
        y=exposure_filtered_df[exposure_columns[1]],
        name=f"Put {exposure_type.title()} Exposure",
        marker_color="rgb(251, 128, 114)",
        hovertemplate="<b>Strike Price</b>: %{x}<br><b>Value</b>: %{y}",
    ))

    fig.add_trace(go.Bar(
        x=exposure_filtered_df["CALLS_Strike"],
        y=exposure_filtered_df[exposure_columns[2]],
        name=f"Net {exposure_type.title()} Exposure",
        marker_color="rgb(255, 255, 179)",
        hovertemplate="<b>Strike Price</b>: %{x}<br><b>Value</b>: %{y}",
        width=0.4
    ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Strike Price",
        yaxis_title=f"{exposure_type.title()} Exposure",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=0, b=20),
        height=400
    )

    exposures = {
        f"Call {exposure_type.title()} Exposure": np.round(np.sum(exposure_filtered_df[exposure_columns[0]]), 2),
        f"Put {exposure_type.title()} Exposure": np.round(np.sum(exposure_filtered_df[exposure_columns[1]]), 2),
        f"Net {exposure_type.title()} Exposure": np.round(np.sum(exposure_filtered_df[exposure_columns[2]]), 2)
    }

    cols = st.columns(len(exposures))
    
    for col, (label, value) in zip(cols, exposures.items()):
        with col:
            with card_container():
                st.metric(label, value)

    with card_container():
        st.plotly_chart(fig, use_container_width=True)

def plot_metric(filtered_df, metric_name, call_column, put_column, x_axis="CALLS_Strike"):
    if metric_name == "IV Skew Vol":
        st.header(metric_name)
    else:
        st.header(metric_name.title())

    metric_df = filtered_df[[x_axis, call_column, put_column]]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=metric_df[x_axis],
        y=metric_df[call_column],
        mode="lines+markers",
        name=f"Call {metric_name}",
        marker_color="rgb(141, 211, 199)",
        hovertemplate=f"<b>Strike Price</b>: %{{x}}<br><b>Value</b>: %{{y}}",
    ))

    fig.add_trace(go.Scatter(
        x=metric_df[x_axis],
        y=metric_df[put_column],
        mode="lines+markers",
        name=f"Put {metric_name}",
        marker_color="rgb(251, 128, 114)",
        hovertemplate=f"<b>Strike Price</b>: %{{x}}<br><b>Value</b>: %{{y}}",
    ))

    fig.update_layout(
        xaxis_title="Strike Price",
        yaxis_title=metric_name,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=0, b=0),
        height=400
    )

    with card_container():
        st.plotly_chart(fig, use_container_width=True)

with st.sidebar:
    st.markdown("<h3>📊 Upload Data</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "label",
        accept_multiple_files=False,
        type=["xls", "xlsx"],
        label_visibility="collapsed"
    )

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = process_data(df)

    with st.sidebar:
        st.markdown("<h3>📅 Select Date</h3>", unsafe_allow_html=True)

        date_selector = st.selectbox(
            "Select Date",
            options=np.sort(df["CALLS_Pos"].unique()),
            index=None,
            format_func=lambda x: x.strftime("%Y-%m-%d") if isinstance(x, pd.Timestamp) else x.strftime("%Y-%m-%d"),
            label_visibility="collapsed"
        )

    if date_selector:
        with st.sidebar:
            filter_zero_values = st.checkbox("Filter out rows with all zero values", value=True)

        filtered_df = df[df["CALLS_Pos"] == date_selector].reset_index(drop=True)
        filtered_df["Row_Number"] = range(len(filtered_df))

        filtered_df["Call_Gamma_Exposure"] = filtered_df["CALLS_Gamma"] * filtered_df["CALLS_Open Int"] * 100
        filtered_df["Put_Gamma_Exposure"] = filtered_df["PUTS_Gamma"] * filtered_df["PUTS_Open Int"] * (-100)
        filtered_df["Net_Gamma_Exposure"] = filtered_df["Call_Gamma_Exposure"] + filtered_df["Put_Gamma_Exposure"]

        filtered_df["Call_Delta_Exposure"] = filtered_df["CALLS_Delta"] * filtered_df["CALLS_Open Int"] * 100
        filtered_df["Put_Delta_Exposure"] = filtered_df["PUTS_Delta"] * filtered_df["PUTS_Open Int"] * 100
        filtered_df["Net_Delta_Exposure"] = filtered_df["Call_Delta_Exposure"] + filtered_df["Put_Delta_Exposure"]

        filtered_df["Call_Hybrid_Exposure"] = filtered_df["Call_Gamma_Exposure"] + filtered_df["Call_Delta_Exposure"]
        filtered_df["Put_Hybrid_Exposure"] = filtered_df["Put_Gamma_Exposure"] + filtered_df["Put_Delta_Exposure"]
        filtered_df["Net_Hybrid_Exposure"] = filtered_df["Call_Hybrid_Exposure"] + filtered_df["Put_Hybrid_Exposure"]

        filtered_df["Call_Vega_Exposure"] = filtered_df["CALLS_Vega"] * filtered_df["CALLS_Open Int"] * 100
        filtered_df["Put_Vega_Exposure"] = filtered_df["PUTS_Vega"] * filtered_df["PUTS_Open Int"] * (-100)
        filtered_df["Net_Vega_Exposure"] = filtered_df["Call_Vega_Exposure"] + filtered_df["Put_Vega_Exposure"]

        plot_exposure(filtered_df, "gamma", filter_zero_values)
        st.divider()
        plot_exposure(filtered_df, "delta", filter_zero_values)
        st.divider()
        plot_exposure(filtered_df, "hybrid", filter_zero_values)
        st.divider()
        plot_exposure(filtered_df, "vega", filter_zero_values)
        st.divider()
        plot_metric(filtered_df, metric_name="Open Interest", call_column="CALLS_Open Int", put_column="PUTS_Open Int")
        st.divider()
        plot_metric(filtered_df, metric_name="IV Skew Vol", call_column="CALLS_IV Skew Vol", put_column="PUTS_IV Skew Vol")

    else:
        st.info("Please select a date.", icon="ℹ️")

else:
    st.info("Please upload your data.", icon="ℹ️")