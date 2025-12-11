# app.py
# Run:
#   python -m streamlit run app.py

import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing


# =============================
# Yahoo Finance safety limits
# =============================
MIN_DATE = date(2000, 1, 1)          # giới hạn sớm nhất cho người dùng chọn
MAX_RANGE_DAYS = 365 * 15            # tối đa 15 năm (để tránh tải quá nặng/dễ lỗi)
TODAY = date.today()


# -----------------------------
# Helpers (robust)
# -----------------------------
def _normalize_colname(x: str) -> str:
    s = str(x).strip().replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([_normalize_colname(p) for p in col if p not in (None, "", "nan")])
            for col in df.columns.to_list()
        ]
    else:
        df = df.copy()
        df.columns = [_normalize_colname(c) for c in df.columns]
    return df


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_colname(c) for c in df.columns]

    found = None
    for c in ("Date", "Datetime"):
        if c in df.columns:
            found = c
            break
    if found is None:
        raise KeyError("Không tìm thấy cột ngày (Date/Datetime).")

    if found != "Date":
        df = df.rename(columns={found: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_prices(ticker: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """
    Tải dữ liệu từ Yahoo Finance trong [start_date, end_date].
    Lưu ý: yfinance end là mốc "exclusive" -> ta truyền end+1 ngày.
    """
    try:
        start_dt = pd.to_datetime(start_date_str)
        end_dt = pd.to_datetime(end_date_str) + pd.Timedelta(days=1)

        raw = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    raw = _flatten_columns(raw)
    df = raw.reset_index()
    df = _ensure_date_column(df)

    # lọc chắc chắn đúng range (inclusive)
    start_dt = pd.to_datetime(start_date_str)
    end_dt_inclusive = pd.to_datetime(end_date_str)
    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt_inclusive)].copy()

    df.columns = [_normalize_colname(c) for c in df.columns]
    return df


def resolve_price_col(df: pd.DataFrame, want_user_label: str, ticker: str) -> str:
    want = _normalize_colname(want_user_label)  # Adj_Close / Close
    tkr = _normalize_colname(ticker.upper())
    cols = [str(c) for c in df.columns]

    if want in cols:
        return want

    cand1 = f"{want}_{tkr}"
    if cand1 in cols:
        return cand1

    cand2 = f"{tkr}_{want}"
    if cand2 in cols:
        return cand2

    matches = [c for c in cols if c == want or c.startswith(want + "_") or c.endswith("_" + want) or (want in c)]
    if matches:
        matches_t = [c for c in matches if tkr in c]
        return matches_t[0] if matches_t else matches[0]

    raise KeyError(f"Không tìm thấy cột giá cho '{want_user_label}'. Một số cột hiện có: {cols[:20]} ...")


def to_frequency(df: pd.DataFrame, price_col: str, freq: str) -> pd.DataFrame:
    if "Date" not in df.columns:
        df = _ensure_date_column(df)
    if price_col not in df.columns:
        raise KeyError(f"Không có cột '{price_col}' trong DataFrame.")

    s = df[["Date", price_col]].dropna().copy()
    s = s.sort_values("Date").set_index("Date")
    s = s[~s.index.duplicated(keep="last")]

    if freq == "D":
        return s.rename(columns={price_col: "y"})
    if freq == "W":
        return s.resample("W-FRI").last().dropna().rename(columns={price_col: "y"})
    if freq == "M":
        return s.resample("M").last().dropna().rename(columns={price_col: "y"})
    raise ValueError("Unsupported freq. Use 'D', 'W', or 'M'.")


def seasonal_periods_for(freq: str) -> int:
    return {"D": 252, "W": 52, "M": 12}[freq]


def iterative_ma_forecast(y: pd.Series, horizon: int, window: int) -> pd.Series:
    hist = list(y.astype(float).values)
    preds = []
    for _ in range(horizon):
        w = hist[-window:] if len(hist) >= window else hist[:]
        nxt = float(np.mean(w)) if len(w) else float("nan")
        preds.append(nxt)
        hist.append(nxt)
    return pd.Series(preds, name="forecast")


def naive_forecast(y: pd.Series, horizon: int) -> pd.Series:
    last = float(y.iloc[-1])
    return pd.Series([last] * horizon, name="forecast")


def make_future_index(last_dt: pd.Timestamp, freq: str, horizon: int) -> pd.DatetimeIndex:
    rule = {"D": "B", "W": "W-FRI", "M": "M"}[freq]
    return pd.date_range(start=last_dt, periods=horizon + 1, freq=rule)[1:]


def forecast_series(
    y: pd.Series,
    method: str,
    horizon: int,
    freq: str,
    alpha_mode: str = "Fixed 0.1",
    alpha_fixed: float = 0.1,
    trend_type: str = "add",
    seasonal_type: str = "add",
    hw_alpha: float = 0.1,
    hw_beta: float = 0.2,
    hw_gamma: float = 0.2,
) -> pd.Series:

    y = y.dropna().astype(float)

    if len(y) < 10:
        raise ValueError("Chuỗi quá ngắn (<10 điểm). Hãy chọn range dài hơn hoặc tần suất khác.")

    if method == "Naive":
        return naive_forecast(y, horizon)

    if method == "Moving Average (3)":
        return iterative_ma_forecast(y, horizon, window=3)

    if method == "Moving Average (6)":
        return iterative_ma_forecast(y, horizon, window=6)

    if method == "Simple Exponential Smoothing":
        if alpha_mode == "Fixed 0.1":
            a = float(alpha_fixed)
            if not (0.0 <= a <= 1.0):
                raise ValueError("alpha phải nằm trong [0, 1].")
            model = SimpleExpSmoothing(y, initialization_method="estimated").fit(
                smoothing_level=a, optimized=False
            )
        else:
            model = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)

        fc = model.forecast(horizon)
        fc.name = "forecast"
        return fc

    if method == "Holt (trend)":
        model = Holt(y, initialization_method="estimated").fit(optimized=True)
        fc = model.forecast(horizon)
        fc.name = "forecast"
        return fc

    if method == "Holt-Winters (trend + seasonality)":
        # mul không dùng được nếu có giá <= 0
        if (trend_type == "mul" or seasonal_type == "mul") and (y.min() <= 0):
            raise ValueError("Mul cần dữ liệu > 0. Hãy chọn trend/seasonal = add.")

        sp = seasonal_periods_for(freq)
        sp = min(sp, max(2, len(y) // 2))  # tránh sp quá lớn so với độ dài chuỗi

        # dùng hệ số do user chọn
        model = ExponentialSmoothing(
            y,
            trend=trend_type,
            seasonal=seasonal_type,
            seasonal_periods=sp,
            initialization_method="estimated",
        ).fit(
            smoothing_level=hw_alpha,        # alpha
            smoothing_trend=hw_beta,         # beta
            smoothing_seasonal=hw_gamma,     # gamma
            optimized=False,
        )

        fc = model.forecast(horizon)
        fc.name = "forecast"
        return fc


    raise ValueError("Unsupported method")


def plot_actual_forecast(
    ts: pd.DataFrame,
    fc: pd.Series,
    freq: str,
    title: str,
    price_label: str = "Price",
    ma_window=None,
) -> pd.Series:
    """
    Vẽ:
    - Giá thực tế (Adj Close / Close) – xanh lá
    - (Nếu chọn) Đường MA – màu vàng/cam
    - Đường dự báo – nét đứt
    """
    y = ts["y"].dropna()
    if y.empty:
        return fc

    # map forecast series sang index thời gian tương lai
    last_dt = y.index[-1]
    future_idx = make_future_index(last_dt, freq, len(fc))
    fc = pd.Series(np.asarray(fc.values, dtype=float), index=future_idx, name="forecast")

    fig = go.Figure()

    # Giá thực tế – xanh lá
    fig.add_trace(
        go.Scatter(
            x=y.index,
            y=y.values,
            mode="lines",
            name=price_label,
            line=dict(color="lime", width=1.2),
        )
    )

    # Đường MA – vàng/cam
    if ma_window is not None and ma_window > 0 and len(y) >= 2:
        ma = y.rolling(window=int(ma_window), min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=ma.index,
                y=ma.values,
                mode="lines",
                name=f"MA{int(ma_window)}",
                line=dict(color="orange", width=2),
            )
        )

    # Forecast – nét đứt
    fig.add_trace(
        go.Scatter(
            x=fc.index,
            y=fc.values,
            mode="lines",
            name="Forecast",
            line=dict(width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=price_label,
        height=520,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    return fc

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Stock Forecast App", layout="wide")
st.markdown(
    """
    <style>
    .watermark-lan {
        position: fixed;
        right: 18px;
        bottom: 14px;
        z-index: 9999;
        font-size: 14px;
        font-weight: 700;
        opacity: 0.45;          /* độ mờ */
        pointer-events: none;   /* không bị click */
        user-select: none;      /* không bôi chọn */
    }
    </style>

    <div class="watermark-lan">Lân</div>
    """,
    unsafe_allow_html=True
)

st.title("📈 Ứng dụng dự báo giá cổ phiếu (Yahoo Finance)")

with st.sidebar:
    st.header("Cấu hình")
    ticker = st.text_input("Mã cổ phiếu (Ticker)", value="AMD").strip().upper()

    # Default range: từ 11/2021 đến hôm nay, nhưng không sớm hơn MIN_DATE
    default_start = max(date(2021, 11, 1), MIN_DATE)
    default_end = TODAY

    st.caption(f"Giới hạn chọn ngày: từ {MIN_DATE.isoformat()} đến {TODAY.isoformat()}")
    start_end = st.date_input(
        "Khoảng dữ liệu (từ - đến)",
        value=(default_start, default_end),
        min_value=MIN_DATE,
        max_value=TODAY,
    )

    price_choice = st.selectbox("Cột giá dùng để dự báo", ["Adj Close", "Close"], index=0)
    freq_label = st.selectbox("Tần suất", ["Ngày", "Tuần", "Tháng"], index=0)
    freq = {"Ngày": "D", "Tuần": "W", "Tháng": "M"}[freq_label]
        # Tuỳ chọn vẽ thêm đường trung bình động (MA) trên biểu đồ
    ma_option = st.selectbox(
        "Đường trung bình động (MA) hiển thị",
        ["Không vẽ MA", "MA20", "MA50", "MA100", "MA200"],
        index=2,  # mặc định MA50
    )
    if ma_option == "Không vẽ MA":
        ma_window = None
    else:
        ma_window = int(ma_option.replace("MA", ""))

    method = st.selectbox(
        "Kỹ thuật dự báo",
        [
            "Naive",
            "Moving Average (3)",
            "Moving Average (6)",
            "Simple Exponential Smoothing",
            "Holt (trend)",
            "Holt-Winters (trend + seasonality)",
        ],
        index=0,
    )

    horizon = st.slider("Số bước dự báo (horizon)", min_value=5, max_value=120, value=20)

    alpha_mode = "Fixed 0.1"
    alpha_fixed = 0.1
    trend_type = "add"
    seasonal_type = "add"
    hw_alpha = 0.1
    hw_beta = 0.2
    hw_gamma = 0.2

    if method == "Simple Exponential Smoothing":
        alpha_mode = st.radio("Chọn alpha", ["Fixed 0.1", "Optimized"], index=0)
        if alpha_mode == "Fixed 0.1":
            alpha_fixed = st.number_input("alpha (0–1)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    if method == "Holt-Winters (trend + seasonality)":
        trend_type = st.selectbox("Trend", ["add", "mul"], index=0)
        seasonal_type = st.selectbox("Seasonal", ["add", "mul"], index=0)

        st.markdown("### Chọn hệ số alpha và beta:")
        hw_alpha = st.slider(
            "Alpha (Smoothing Level):",
            min_value=0.01,
            max_value=1.0,
            value=0.10,
            step=0.01,
        )
        hw_beta = st.slider(
            "Beta (Smoothing Trend):",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
        )
        hw_gamma = st.slider(
            "Gamma (Smoothing Seasonality):",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
        )

    run = st.button("Chạy dự báo")



# -----------------------------
# Parse & enforce date range
# -----------------------------
def _parse_date_input(x):
    # streamlit date_input có thể trả 1 date hoặc tuple(date, date)
    if isinstance(x, tuple) or isinstance(x, list):
        if len(x) == 2:
            return x[0], x[1]
        return x[0], x[0]
    return x, x

start_d, end_d = _parse_date_input(start_end)

# đảm bảo thứ tự
if start_d > end_d:
    start_d, end_d = end_d, start_d

# clamp to allowed range
start_d = max(start_d, MIN_DATE)
end_d = min(end_d, TODAY)

# enforce max span
span_days = (end_d - start_d).days
if span_days > MAX_RANGE_DAYS:
    # tự động co lại để không lỗi / không tải quá nặng
    start_d = end_d - timedelta(days=MAX_RANGE_DAYS)
    st.sidebar.warning(
        f"Khoảng thời gian quá dài. App tự giới hạn tối đa {MAX_RANGE_DAYS//365} năm "
        f"(từ {start_d.isoformat()} đến {end_d.isoformat()})."
    )

start_str = start_d.isoformat()
end_str = end_d.isoformat()


# -----------------------------
# Run
# -----------------------------
if run:
    with st.spinner("Đang tải dữ liệu từ Yahoo Finance..."):
        df = fetch_prices(ticker, start_date_str=start_str, end_date_str=end_str)

    if df.empty:
        st.error("Không tải được dữ liệu. Kiểm tra ticker hoặc thử đổi khoảng ngày.")
        st.stop()

    # resolve cột giá (ưu tiên user chọn, fallback Close)
    try:
        price_col = resolve_price_col(df, price_choice, ticker)
    except Exception:
        st.warning(f"Không có cột '{price_choice}'. Chuyển sang 'Close'.")
        price_col = resolve_price_col(df, "Close", ticker)

    # series theo tần suất
    try:
        ts = to_frequency(df, price_col=price_col, freq=freq)
    except Exception as e:
        st.error(f"Lỗi khi chuẩn hoá tần suất dữ liệu: {e}")
        st.stop()

    st.subheader("Dữ liệu sau khi chuẩn hóa tần suất")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(ts.tail(15), use_container_width=True)
    with c2:
        # st.metric không nhận datetime.date -> dùng string
        st.metric("Số quan sát", f"{len(ts)}")
        st.metric("Từ ngày", ts.index.min().strftime("%Y-%m-%d"))
        st.metric("Đến ngày", ts.index.max().strftime("%Y-%m-%d"))

    # forecas
    try:
        fc = forecast_series(
            y=ts["y"],
            method=method,
            horizon=horizon,
            freq=freq,
            alpha_mode=alpha_mode,
            alpha_fixed=alpha_fixed,
            trend_type=trend_type,
            seasonal_type=seasonal_type,
            hw_alpha=hw_alpha,
            hw_beta=hw_beta,
            hw_gamma=hw_gamma,
        )

    except Exception as e:
        st.error(f"Lỗi khi dự báo: {e}")
        st.stop()

    st.subheader("Biểu đồ dự báo")
    fc_indexed = plot_actual_forecast(
        ts,
        fc,
        freq,
        title=f"{ticker} | {freq_label} | {method} | Horizon={horizon} | Range={start_str}→{end_str}",
        price_label=price_choice,   # Adj Close / Close
        ma_window=ma_window,        # MA20/50/100/200 hoặc None
    )

    st.subheader("Bảng dự báo")
    out = pd.DataFrame({"forecast": fc_indexed})
    st.dataframe(out, use_container_width=True)

else:
    st.info("Chọn cấu hình ở sidebar và bấm **Chạy dự báo**.")



