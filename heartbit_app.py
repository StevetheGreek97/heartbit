import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# Utils
# =========================
def rfft_spectrum(x: np.ndarray, fps: float):
    x = np.asarray(x, float)
    N = len(x)
    if N < 2:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])
    dt = 1.0 / fps
    X = np.fft.rfft(x)
    f_hz = np.fft.rfftfreq(N, d=dt)
    amp = np.abs(X) / N
    if N > 2:
        amp[1:-1] *= 2
    return f_hz, X, amp

def find_peak_in_band(f_hz, amp, band):
    lo, hi = band
    mask = (f_hz >= lo) & (f_hz <= hi)
    if not np.any(mask):
        raise ValueError("No frequency bins in the specified search range.")
    return float(f_hz[mask][np.argmax(amp[mask])])

def sliding_window_hann(signal, window_s, overlap_s, fps, band):
    overlap = int(overlap_s * fps)
    window  = int(window_s * fps)
    step    = window - overlap
    if window <= 0:
        raise ValueError("Window must be > 0 seconds.")
    if step <= 0:
        raise ValueError("Overlap must be smaller than window length.")
    hann = np.hanning(window)
    bpm, centers = [], []
    for i in range(0, len(signal) - window + 1, step):
        frame = signal[i:i+window] * hann
        f_hz, X, amp = rfft_spectrum(frame, fps)
        f_peak = find_peak_in_band(f_hz, amp, band)
        bpm.append(f_peak * 60.0)
        centers.append((i + window/2) / fps)
    return np.array(bpm), np.array(centers)

def hampel_mask(x, window=5, n_sigmas=3):
    x = np.asarray(x, float)
    n = len(x)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i-window); hi = min(n, i+window+1)
        seg = x[lo:hi]
        med = np.nanmedian(seg)
        mad = np.nanmedian(np.abs(seg - med)) or 1e-9
        if np.abs(x[i] - med) > n_sigmas * 1.4826 * mad:
            mask[i] = True
    y = x.copy(); y[mask] = np.nan
    idx = np.arange(n); good = ~np.isnan(y)
    if np.any(good):
        y[~good] = np.interp(idx[~good], idx[good], y[good])
    else:
        y[:] = np.nanmedian(x)
    return y

# =========================
# App
# =========================
st.set_page_config(page_title="Daphnia BPM Analyzer", layout="wide")
st.title("Daphnia BPM Analyzer")
st.caption("Upload CSV → pick signal → get spectrum, filtered trace, and BPM over time.")

with st.sidebar:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    fps = st.number_input("FPS", min_value=1.0, max_value=10000.0, value=60.0, step=1.0)
    lo = st.number_input("Min freq (Hz)", min_value=0.0, value=0.5, step=0.1)
    hi = st.number_input("Max freq (Hz)", min_value=0.1, value=10.0, step=0.1)
    window_s = st.number_input("Window (s)", min_value=0.5, value=8.0, step=0.5)
    overlap_s = st.number_input("Overlap (s)", min_value=0.0, value=2.0, step=0.5)
    bandwidth = st.number_input("Half-bandwidth (Hz)", min_value=0.1, value=2.0, step=0.1)
    use_hampel = st.checkbox("Hampel clean BPM", value=True)
    hampel_win = st.number_input("Hampel window", min_value=1, value=5, step=1)
    hampel_sig = st.number_input("Hampel nσ", min_value=0.5, value=3.0, step=0.5)

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# Read CSV (comma-separated)
df = pd.read_csv(uploaded)

# Pick signal (default to area_px if present)
cols = list(df.columns)
default_idx = cols.index("area_px") if "area_px" in cols else 0
col_signal = st.selectbox("Signal column", options=cols, index=default_idx)

# Coerce to numeric + interpolate
sig_series = pd.to_numeric(df[col_signal], errors="coerce").astype(float)
if sig_series.isna().all():
    st.error(f"Column '{col_signal}' is all NaN after numeric coercion.")
    st.stop()
sig = sig_series.to_numpy()
idx = np.arange(len(sig))
good = ~np.isnan(sig)
sig[~good] = np.interp(idx[~good], idx[good], sig[good])

N = len(sig)
t = np.arange(N) / fps
band = (lo, hi)

# Spectrum + peak
f_hz, X, amp = rfft_spectrum(sig, fps)
try:
    f_peak = find_peak_in_band(f_hz, amp, band)
except ValueError as e:
    st.error(str(e))
    st.stop()
bpm_peak = f_peak * 60.0

# Peak-centered "band-pass" via freq mask
bp_low = max(lo, f_peak - bandwidth)
bp_high = min(hi, f_peak + bandwidth)
BP = ((f_hz >= bp_low) & (f_hz <= bp_high)).astype(float)
sig_filt = np.fft.irfft(X * BP, n=N)

# --------- Plotly: stacked raw + filtered ---------
fig_tf = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Raw signal", "Filtered (peak ± bandwidth)")
)

fig_tf.add_trace(
    go.Scatter(x=t, y=sig, mode="lines", name="Raw"),
    row=1, col=1
)
fig_tf.add_trace(
    go.Scatter(x=t, y=sig_filt, mode="lines", name="Filtered"),
    row=2, col=1
)
fig_tf.update_xaxes(title_text="Time (s)", row=2, col=1)
fig_tf.update_yaxes(title_text=col_signal, row=1, col=1)
fig_tf.update_yaxes(title_text="Filtered", row=2, col=1)
fig_tf.update_layout(height=600, margin=dict(l=40, r=20, t=60, b=40), showlegend=False)
st.plotly_chart(fig_tf, use_container_width=True)

# --------- Plotly: spectrum with peak marker ---------
fig_spec = go.Figure()
fig_spec.add_trace(go.Scatter(x=f_hz, y=amp, mode="lines", name="Amplitude"))
fig_spec.add_vline(x=f_peak, line_dash="dash", annotation_text=f"{f_peak:.2f} Hz ({bpm_peak:.0f} BPM)", annotation_position="top right")
fig_spec.update_xaxes(title_text="Frequency (Hz)", range=[lo, hi])
fig_spec.update_yaxes(title_text="Amplitude")
fig_spec.update_layout(height=350, margin=dict(l=40, r=20, t=40, b=40))
st.plotly_chart(fig_spec, use_container_width=True)

# --------- Sliding window BPM ---------
bpm_rate, centers = sliding_window_hann(sig, window_s, overlap_s, fps, band)
bpm_clean = hampel_mask(bpm_rate, window=int(hampel_win), n_sigmas=float(hampel_sig)) if use_hampel else bpm_rate

fig_bpm = go.Figure()
fig_bpm.add_trace(go.Scatter(x=centers, y=bpm_rate, mode="lines", name="BPM"))
if use_hampel:
    fig_bpm.add_trace(go.Scatter(x=centers, y=bpm_clean, mode="lines", name="BPM (Hampel)"))
fig_bpm.update_xaxes(title_text="Time (s)")
fig_bpm.update_yaxes(title_text="BPM")
fig_bpm.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
st.plotly_chart(fig_bpm, use_container_width=True)

# --------- Download ---------
out_df = pd.DataFrame({
    "center_time_s": centers,
    "bpm": bpm_rate,
    "bpm_clean": bpm_clean if use_hampel else np.nan
})
import zipfile
import io

# Save plots as HTML
buf_time = io.StringIO()
fig_tf.write_html(buf_time)
buf_spec = io.StringIO()
fig_spec.write_html(buf_spec)
buf_bpm = io.StringIO()
fig_bpm.write_html(buf_bpm)

# Save CSV
csv_bytes = out_df.to_csv(index=False).encode("utf-8")

# Create in-memory ZIP
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("bpm_by_window.csv", csv_bytes)
    zf.writestr("time_series.html", buf_time.getvalue())
    zf.writestr("spectrum.html", buf_spec.getvalue())
    zf.writestr("bpm_over_time.html", buf_bpm.getvalue())

# Download button
st.download_button(
    label="Download results (CSV + plots)",
    data=zip_buffer.getvalue(),
    file_name="daphnia_bpm_results.zip",
    mime="application/zip"
)

# Quick summary
st.success(f"Peak: {f_peak:.2f} Hz  •  {bpm_peak:.0f} BPM  •  Band-pass: [{bp_low:.2f}, {bp_high:.2f}] Hz")
