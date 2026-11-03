from __future__ import annotations

import csv
import io
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# LearnWise Universal CSV Studio
# Requirements: streamlit, pandas, numpy
# Run: streamlit run app.py
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="LearnWise | CSV Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "3.0"


# ------------------------------- Styling -------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f5f7fc; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151b35 0%, #25345d 100%);
    }
    [data-testid="stSidebar"] * { color: #f5f7ff; }
    .block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.035em; }
    .hero {
        padding: 30px 34px; border-radius: 22px; color: white;
        background: linear-gradient(115deg, #202653, #5145b8 58%, #21829b);
        margin-bottom: 22px; box-shadow: 0 14px 38px rgba(31, 42, 93, .15);
    }
    .hero .eyebrow { color: #dce4ff; font-size: .72rem; font-weight: 800; letter-spacing: .15em; }
    .hero h1 { color: white; font-size: clamp(2rem, 4vw, 3rem); margin: .45rem 0; }
    .hero p { color: #edf1ff; margin: 0; max-width: 850px; }
    .metric-card {
        background: white; border: 1px solid #e7eaf2; border-radius: 16px;
        padding: 17px 19px; min-height: 104px; box-shadow: 0 5px 18px rgba(25, 35, 75, .04);
    }
    .metric-label { color: #6d7690; font-size: .78rem; font-weight: 700; }
    .metric-value { color: #20294b; font: 800 1.8rem 'Manrope', sans-serif; margin-top: 7px; }
    div[data-testid="stDataFrame"] { border: 1px solid #e4e8f1; border-radius: 12px; }
    div.stButton > button, div.stDownloadButton > button { border-radius: 10px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------ CSV engine ------------------------------

DELIMITER_OPTIONS = {
    "Auto detect": None,
    "Comma (,)": ",",
    "Semicolon (;)": ";",
    "Tab": "\t",
    "Pipe (|)": "|",
}


def _delimiter_score(text: str, delimiter: str) -> tuple[int, float]:
    """Score a delimiter by how consistently it splits the first non-empty rows."""
    try:
        rows = [
            row for row in csv.reader(io.StringIO(text[:200_000]), delimiter=delimiter)
            if row and any(str(cell).strip() for cell in row)
        ][:30]
        if len(rows) < 2:
            return (0, 0.0)
        widths = [len(row) for row in rows]
        width = max(set(widths), key=widths.count)
        consistency = sum(n == width for n in widths) / len(widths)
        return (width if consistency >= 0.5 else 0, consistency)
    except Exception:
        return (0, 0.0)


def _choose_delimiter(text: str) -> str:
    scores = [(d, _delimiter_score(text, d)) for d in [",", ";", "\t", "|"]]
    scores.sort(key=lambda item: (item[1][0], item[1][1]), reverse=True)
    return scores[0][0] if scores and scores[0][1][0] > 1 else ","


def load_csv(
    uploaded_file,
    delimiter_choice: str = "Auto detect",
    header_row: bool = True,
    skip_bad_lines: bool = False,
) -> tuple[pd.DataFrame, str, str, str]:
    """Read a CSV using several common encodings and separator strategies."""
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The selected file is empty.")

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    errors = []
    explicit_delimiter = DELIMITER_OPTIONS.get(delimiter_choice)

    for encoding in encodings:
        try:
            sample = raw[:250_000].decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
            continue

        delimiter = explicit_delimiter or _choose_delimiter(sample)
        separators = [delimiter]
        if explicit_delimiter is None:
            separators = list(dict.fromkeys([delimiter, ",", ";", "\t", "|"]))

        for sep in separators:
            try:
                frame = pd.read_csv(
                    io.BytesIO(raw),
                    encoding=encoding,
                    sep=sep,
                    engine="python",
                    header=0 if header_row else None,
                    on_bad_lines="skip" if skip_bad_lines else "error",
                    skipinitialspace=True,
                    low_memory=False,
                )
                if frame.shape[1] == 1 and explicit_delimiter is None:
                    continue
                if frame.empty and frame.shape[1] == 0:
                    continue
                return frame, encoding, ("TAB" if sep == "\t" else sep), "standard"
            except Exception as exc:
                errors.append(f"{encoding}/{repr(sep)}: {exc}")

        # Last attempt: pandas delimiter inference.
        if explicit_delimiter is None:
            try:
                frame = pd.read_csv(
                    io.BytesIO(raw),
                    encoding=encoding,
                    sep=None,
                    engine="python",
                    header=0 if header_row else None,
                    on_bad_lines="skip" if skip_bad_lines else "error",
                    skipinitialspace=True,
                )
                if frame.shape[1] > 1:
                    return frame, encoding, "Auto inferred", "inferred"
            except Exception as exc:
                errors.append(f"{encoding}/auto: {exc}")

    details = "\n".join(errors[-4:])
    raise ValueError(
        "The file could not be parsed as a consistent CSV. "
        "Try selecting the separator manually or enabling 'Skip malformed rows'.\n\n"
        + details
    )


def clean_headers(frame: pd.DataFrame) -> pd.DataFrame:
    """Create readable, unique column names without changing cell values."""
    result = frame.copy()
    names = []
    seen = {}
    for index, value in enumerate(result.columns):
        name = str(value).replace("\ufeff", "").strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"Column_{index + 1}"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        names.append(name)
    result.columns = names
    return result


def infer_types(frame: pd.DataFrame):
    """Infer numeric/date fields carefully while retaining original text otherwise."""
    data = frame.copy()
    numeric_cols, date_cols = [], []

    for col in data.columns:
        series = data[col]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        values = series.dropna()
        if values.empty:
            continue

        text = values.astype(str).str.strip()
        cleaned = (
            text.str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(r"^\$", "", regex=True)
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")

        # Only convert if nearly all non-empty values are genuinely numeric.
        if numeric.notna().mean() >= 0.95:
            full = (
                series.astype("string").str.strip()
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace(r"^\$", "", regex=True)
            )
            data[col] = pd.to_numeric(full, errors="coerce")
            numeric_cols.append(col)
            continue

        # Avoid interpreting arbitrary identifiers as dates.
        if len(values) >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(values, errors="coerce")
            if parsed.notna().mean() >= 0.95:
                data[col] = pd.to_datetime(series, errors="coerce")
                date_cols.append(col)

    categorical_cols = [
        col for col in data.columns
        if col not in numeric_cols and col not in date_cols
    ]
    return data, numeric_cols, categorical_cols, date_cols


def metric_card(label: str, value: str):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def safe_download(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def category_counts(series: pd.Series, limit: int = 20) -> pd.Series:
    return (
        series.astype("string")
        .fillna("(Missing)")
        .value_counts()
        .head(limit)
    )


# --------------------------------- Header ---------------------------------

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">LEARNWISE · UNIVERSAL CSV STUDIO</div>
      <h1>Explore any dataset with confidence.</h1>
      <p>Upload a CSV with your own column names and data types. Inspect records,
      search and filter rows, create charts, review data quality and export results.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------- Sidebar --------------------------------

with st.sidebar:
    st.markdown("# 📊 LearnWise")
    st.caption("UNIVERSAL DATA INTELLIGENCE")
    st.caption(f"CSV STUDIO · VERSION {APP_VERSION}")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Data explorer",
            "Visual analytics",
            "Column insights",
            "Data quality",
        ],
    )

    st.divider()
    st.markdown("### Import your dataset")
    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv", "txt"],
        help="CSV files with comma, semicolon, tab or pipe separators are supported.",
        key="learnwise_csv_upload",
    )
    delimiter_choice = st.selectbox(
        "Separator",
        list(DELIMITER_OPTIONS.keys()),
        index=0,
        help="Use Auto detect first. Choose a separator manually if columns are not split correctly.",
    )
    header_row = st.checkbox("First row contains column names", value=True)
    skip_bad_lines = st.checkbox(
        "Skip malformed rows",
        value=False,
        help="Use only if the file has inconsistent rows. Some records may be skipped.",
    )
    st.divider()
    st.caption("Supported: common CSV encodings and flexible column names.")


if uploaded is None:
    st.info("Upload a CSV file using the import panel in the sidebar to start.")
    st.markdown(
        """
        **You can use datasets containing any kind of attributes**, for example:
        - Student or employee records
        - Sales, orders, products and customer data
        - Survey responses and research data
        - Weather, finance, inventory or operational data

        No fixed column names or predefined template is required.
        """
    )
    st.stop()


# --------------------------------- Load data ------------------------------

try:
    with st.spinner("Reading and preparing your CSV..."):
        raw_df, used_encoding, used_separator, parse_mode = load_csv(
            uploaded,
            delimiter_choice=delimiter_choice,
            header_row=header_row,
            skip_bad_lines=skip_bad_lines,
        )
except Exception as exc:
    st.error("CSV import failed.")
    st.exception(exc)
    st.info(
        "Check that the file is a CSV, then try Auto detect or select its separator "
        "manually in the sidebar."
    )
    st.stop()

df = clean_headers(raw_df)
df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

if df.empty or len(df.columns) == 0:
    st.error("The file was read, but it contains no usable rows or columns.")
    st.stop()

df, numeric_cols, categorical_cols, date_cols = infer_types(df)
all_columns = df.columns.tolist()

with st.expander("Import details", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.write(f"**File:** {uploaded.name}")
    c2.write(f"**Encoding:** {used_encoding}")
    c3.write(f"**Separator:** {used_separator}")
    c4.write(f"**Rows loaded:** {len(df):,}")
    st.caption(
        "If 'Skip malformed rows' is enabled, inconsistent records may have been skipped."
    )

st.caption(
    f"**{uploaded.name}**  ·  {len(df):,} rows  ·  {len(all_columns):,} columns  ·  "
    f"{len(numeric_cols)} numeric  ·  {len(categorical_cols)} text/category  ·  "
    f"{len(date_cols)} date"
)


# --------------------------------- Overview --------------------------------

if page == "Overview":
    st.subheader("Dataset overview")

    cells = [
        ("Total rows", f"{len(df):,}"),
        ("Total columns", f"{len(all_columns):,}"),
        ("Numeric columns", f"{len(numeric_cols):,}"),
        ("Missing cells", f"{int(df.isna().sum().sum()):,}"),
    ]
    metric_cols = st.columns(4)
    for container, (label, value) in zip(metric_cols, cells):
        with container:
            metric_card(label, value)

    st.markdown("### Data preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Missing values by column")
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0].head(20)
        if not missing.empty:
            st.bar_chart(missing)
        else:
            st.success("No missing values detected.")

    with right:
        st.markdown("#### Explore a numeric distribution")
        if numeric_cols:
            selected = st.selectbox(
                "Choose a numeric column",
                numeric_cols,
                key="overview_numeric",
            )
            values = pd.to_numeric(df[selected], errors="coerce").dropna()
            if not values.empty:
                bin_count = min(20, max(5, int(values.nunique())))

                histogram = pd.cut(
                    values,
                    bins=bin_count,
                    duplicates="drop"
                ).value_counts().sort_index()

                histogram.index = histogram.index.astype(str)

                st.bar_chart(histogram)
            else:
                st.info("This column has no usable numeric values.")
        else:
            st.info("No numeric columns were detected.")

    if categorical_cols:
        st.markdown("#### Category distribution")
        selected_cat = st.selectbox(
            "Choose a text/category column",
            categorical_cols,
            key="overview_category",
        )
        st.bar_chart(category_counts(df[selected_cat]))


# --------------------------------- Explorer --------------------------------

elif page == "Data explorer":
    st.subheader("Search, filter and export records")

    result = df.copy()
    search_text = st.text_input(
        "Search across all columns",
        placeholder="Type a name, ID, category, number or any value...",
    )
    if search_text.strip():
        search_mask = result.astype("string").apply(
            lambda column: column.str.contains(
                search_text.strip(), case=False, na=False, regex=False
            )
        ).any(axis=1)
        result = result[search_mask]

    with st.expander("Filter records", expanded=True):
        selected_filters = st.multiselect(
            "Select columns to filter",
            all_columns,
            help="You can apply multiple filters together.",
        )

        for col in selected_filters:
            if col in numeric_cols:
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if values.empty:
                    st.caption(f"{col}: no numeric values to filter.")
                    continue
                minimum, maximum = float(values.min()), float(values.max())
                if minimum < maximum:
                    selected_range = st.slider(
                        f"{col} range",
                        min_value=minimum,
                        max_value=maximum,
                        value=(minimum, maximum),
                        key=f"filter_numeric_{col}",
                    )
                    result = result[
                        pd.to_numeric(result[col], errors="coerce").between(
                            selected_range[0], selected_range[1]
                        )
                    ]

            elif col in date_cols:
                values = pd.to_datetime(df[col], errors="coerce").dropna()
                if values.empty:
                    continue
                selected_dates = st.date_input(
                    f"{col} date range",
                    value=(values.min().date(), values.max().date()),
                    key=f"filter_date_{col}",
                )
                if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
                    dates = pd.to_datetime(result[col], errors="coerce")
                    start = pd.Timestamp(selected_dates[0])
                    end = pd.Timestamp(selected_dates[1]) + pd.Timedelta(days=1)
                    result = result[dates.ge(start) & dates.lt(end)]

            else:
                options = (
                    df[col].dropna().astype(str).value_counts().head(500).index.tolist()
                )
                chosen_values = st.multiselect(
                    f"{col} values",
                    options,
                    key=f"filter_category_{col}",
                )
                if chosen_values:
                    result = result[result[col].astype(str).isin(chosen_values)]

    st.write(f"**{len(result):,} matching rows** out of {len(df):,}")
    st.dataframe(result, use_container_width=True, hide_index=True, height=520)

    st.download_button(
        "⬇ Download filtered CSV",
        data=safe_download(result),
        file_name="learnwise_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


# --------------------------------- Visual analytics ------------------------

elif page == "Visual analytics":
    st.subheader("Create charts from your dataset")
    chart_type = st.selectbox(
        "Chart type",
        ["Bar chart", "Line chart", "Histogram", "Scatter plot", "Category frequency"],
    )

    if chart_type == "Histogram":
        if not numeric_cols:
            st.warning("A histogram needs at least one numeric column.")
        else:
            column = st.selectbox("Numeric column", numeric_cols)
            bins = st.slider("Number of bins", 5, 60, 20)
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if not values.empty:
                counts = pd.cut(values, bins=bins, duplicates="drop").value_counts().sort_index()
                counts.index = counts.index.astype(str)
                st.bar_chart(counts)
            else:
                st.info("No numeric values are available in this column.")

    elif chart_type in ("Bar chart", "Line chart"):
        if not numeric_cols:
            st.warning("Bar and line charts need at least one numeric column.")
        else:
            x_column = st.selectbox("X axis (category, date or other field)", all_columns)
            y_column = st.selectbox("Y axis (numeric field)", numeric_cols)
            aggregation = st.selectbox(
                "Aggregation",
                ["Mean", "Sum", "Count", "Minimum", "Maximum"],
            )

            chart_data = df[[x_column, y_column]].copy()
            chart_data[y_column] = pd.to_numeric(chart_data[y_column], errors="coerce")
            chart_data = chart_data.dropna(subset=[x_column, y_column])

            if not chart_data.empty:
                if chart_data[x_column].nunique() > 100:
                    st.warning("The X column has many unique values. Showing the 100 most common.")
                    top_values = chart_data[x_column].value_counts().head(100).index
                    chart_data = chart_data[chart_data[x_column].isin(top_values)]

                grouped = chart_data.groupby(x_column, dropna=False)[y_column]
                if aggregation == "Mean":
                    plotted = grouped.mean()
                elif aggregation == "Sum":
                    plotted = grouped.sum()
                elif aggregation == "Count":
                    plotted = grouped.count()
                elif aggregation == "Minimum":
                    plotted = grouped.min()
                else:
                    plotted = grouped.max()

                plotted = plotted.sort_values(ascending=False) if chart_type == "Bar chart" else plotted.sort_index()
                st.caption(f"{aggregation} of {y_column} grouped by {x_column}")
                if chart_type == "Bar chart":
                    st.bar_chart(plotted)
                else:
                    st.line_chart(plotted)
            else:
                st.info("No valid records are available for the selected fields.")

    elif chart_type == "Scatter plot":
        if len(numeric_cols) < 2:
            st.warning("A scatter plot needs at least two numeric columns.")
        else:
            left, right = st.columns(2)
            x_column = left.selectbox("X axis", numeric_cols, key="scatter_x")
            y_options = [column for column in numeric_cols if column != x_column]
            y_column = right.selectbox("Y axis", y_options, key="scatter_y")
            points = df[[x_column, y_column]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if len(points) > 5000:
                points = points.sample(5000, random_state=42)
                st.caption("Showing a random sample of 5,000 rows for performance.")
            if not points.empty:
                st.scatter_chart(points, x=x_column, y=y_column, use_container_width=True)
            else:
                st.info("No valid numeric pairs were found.")

    else:
        if not categorical_cols:
            st.warning("No text/category columns were detected.")
        else:
            column = st.selectbox("Category column", categorical_cols)
            limit = st.slider("Number of categories", 5, 50, 15)
            st.bar_chart(category_counts(df[column], limit=limit))


# --------------------------------- Column insights -------------------------

elif page == "Column insights":
    st.subheader("Understand an individual column")
    selected_column = st.selectbox("Choose a column", all_columns)
    series = df[selected_column]

    a, b, c, d = st.columns(4)
    a.metric("Non-empty", f"{series.notna().sum():,}")
    b.metric("Missing", f"{series.isna().sum():,}")
    c.metric("Unique values", f"{series.nunique(dropna=True):,}")
    d.metric("Data type", str(series.dtype))

    if selected_column in numeric_cols:
        values = pd.to_numeric(series, errors="coerce")
        st.markdown("#### Statistical summary")
        st.dataframe(values.describe().to_frame("Value"), use_container_width=True)
        valid = values.dropna()
        if not valid.empty:
            bins = st.slider("Histogram bins", 5, 60, 20)
            counts = pd.cut(valid, bins=bins, duplicates="drop").value_counts().sort_index()
            counts.index = counts.index.astype(str)
            st.bar_chart(counts)

    elif selected_column in date_cols:
        values = pd.to_datetime(series, errors="coerce").dropna()
        if not values.empty:
            st.write(f"**Earliest:** {values.min()}  |  **Latest:** {values.max()}")
            monthly = values.dt.to_period("M").astype(str).value_counts().sort_index()
            st.markdown("#### Records by month")
            st.bar_chart(monthly)
        else:
            st.info("No valid dates were found.")

    else:
        counts = category_counts(series, limit=50)
        st.markdown("#### Most frequent values")
        st.bar_chart(counts)
        st.dataframe(counts.rename("Count").to_frame(), use_container_width=True)


# --------------------------------- Data quality ----------------------------

elif page == "Data quality":
    st.subheader("Data quality report")

    missing_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    complete_count = total_cells - missing_count

    a, b, c, d = st.columns(4)
    a.metric("Missing cells", f"{missing_count:,}")
    b.metric("Duplicate rows", f"{duplicate_count:,}")
    c.metric("Complete cells", f"{complete_count:,}")
    d.metric("Completeness", f"{(complete_count / total_cells * 100):.1f}%" if total_cells else "N/A")

    report = pd.DataFrame(
        {
            "Column": all_columns,
            "Detected type": [str(df[column].dtype) for column in all_columns],
            "Non-empty": [int(df[column].notna().sum()) for column in all_columns],
            "Missing": [int(df[column].isna().sum()) for column in all_columns],
            "Missing (%)": [
                round(float(df[column].isna().mean() * 100), 2)
                for column in all_columns
            ],
            "Unique": [int(df[column].nunique(dropna=True)) for column in all_columns],
        }
    )

    st.markdown("#### Column quality")
    st.dataframe(report, use_container_width=True, hide_index=True)

    missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_percent = missing_percent[missing_percent > 0]
    if not missing_percent.empty:
        st.markdown("#### Missing data percentage")
        st.bar_chart(missing_percent)
    else:
        st.success("No missing values detected.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Duplicate row handling")
        st.write(f"Found **{duplicate_count:,}** fully duplicated rows.")
        if duplicate_count:
            if st.button("Create duplicate-free export", use_container_width=True):
                deduplicated = df.drop_duplicates()
                st.session_state["learnwise_deduplicated"] = deduplicated
                st.success(f"Prepared {len(deduplicated):,} unique rows.")
        if "learnwise_deduplicated" in st.session_state:
            st.download_button(
                "⬇ Download duplicate-free CSV",
                data=safe_download(st.session_state["learnwise_deduplicated"]),
                file_name="learnwise_deduplicated.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with right:
        st.markdown("#### Export current dataset")
        st.download_button(
            "⬇ Download loaded dataset",
            data=safe_download(df),
            file_name="learnwise_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )


# --------------------------------- Footer ----------------------------------

st.divider()
st.caption(
    f"LearnWise CSV Studio v{APP_VERSION} · Flexible CSV import · "
    "Exploration · Visualization · Data quality"
)
