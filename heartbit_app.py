import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from scipy.fft import rfft, rfftfreq
import plotly.graph_objects as go

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="Heartbeat Analyzer", layout="wide")
st.title("🧠 Heartbeat Signal Analyzer")

# ------------------ HELPERS ------------------ #
def compute_fft(signal, fs):
    N = len(signal)
    yf = rfft(signal)
    xf = rfftfreq(N, 1 / fs)
    return xf, np.abs(yf)

def compute_fft_bpm(signal, fs):
    xf, yf = compute_fft(signal, fs)
    dominant_freq = xf[np.argmax(yf)]
    return dominant_freq * 60

def compute_peak_bpm(signal, fs, min_bpm=200, max_bpm=700, prominence=0.1, fallback_if_invalid=True):
    try:
        min_interval_sec = 60 / max_bpm
        max_interval_sec = 60 / min_bpm
        min_distance = int(min_interval_sec * fs)

        peaks, _ = find_peaks(signal, distance=min_distance, prominence=prominence)

        if len(peaks) < 2:
            return np.nan

        intervals = np.diff(peaks) / fs
        valid_intervals = intervals[(intervals >= min_interval_sec) & (intervals <= max_interval_sec)]

        if len(valid_intervals) > 0:
            avg_interval = np.mean(valid_intervals)
        elif fallback_if_invalid:
            avg_interval = np.mean(intervals)
        else:
            return np.nan

        bpm = 60 / avg_interval
        return bpm if np.isfinite(bpm) and bpm > 0 else np.nan

    except Exception as e:
        # Optional: print or log the error
        # print(f"Error in compute_peak_bpm: {e}")
        return np.nan


def sliding_window_bpm(signal, time, fs, window_sec, step_sec, min_bpm=200, max_bpm=600, prominence=0.1):
    window_size = int(window_sec * fs)
    step_size = int(step_sec * fs)
    centers, bpm_fft_list, bpm_peak_list = [], [], []

    for start in range(0, len(signal) - window_size, step_size):
        end = start + window_size
        win_sig = signal[start:end]
        win_time = time[start:end]

        bpm_fft = compute_fft_bpm(win_sig, fs)
        bpm_peak = compute_peak_bpm(win_sig, fs, min_bpm, max_bpm, prominence)
        center_time = win_time[len(win_time) // 2]

        centers.append(center_time)
        bpm_fft_list.append(bpm_fft)
        bpm_peak_list.append(bpm_peak)

    return centers, bpm_peak_list, bpm_fft_list

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def find_fft_peak(signal, fs):
    xf, yf = compute_fft(signal, fs)
    idx_peak = np.argmax(yf)
    return xf, yf, xf[idx_peak], yf[idx_peak]

# ------------------ MAIN APP ------------------ #
def main():
    uploaded_file = st.sidebar.file_uploader("📁 Step 1: Upload CSV", type=["csv"])

    if not uploaded_file:
        st.info("📄 Upload a CSV file to begin analysis.")
        return

    df = pd.read_csv(uploaded_file)
    column = st.sidebar.selectbox("📈 Step 2: Choose Signal Column", df.columns)
    signal = df[column].values
    time = np.arange(len(signal))

    fps = st.sidebar.number_input("🎞️ Step 3: Frame Rate (FPS)", min_value=1, value=30)
    time = time / fps

    with st.sidebar.expander("🎚️ Step 4: Filtering Options", expanded=False):
        apply_band = st.checkbox("Apply Bandpass Filter", value=True)
        lowcut = st.slider("Lowcut Frequency (Hz)", 0.1, 5.0, 3.0) if apply_band else None
        highcut = st.slider("Highcut Frequency (Hz)", 1.0, 12.0, 10.0) if apply_band else None
        order = st.slider("Filter Order", 1, 10, 2) if apply_band else None

        apply_savgol = st.checkbox("Apply Savitzky-Golay Smoothing", value=True)
        window_length = st.slider("Savitzky-Golay Window Length", 5, 99, 9, step=2) if apply_savgol else None
        polyorder = st.slider("Polynomial Order", 1, 5, 3) if apply_savgol else None

    with st.sidebar.expander("📌 Step 5: Peak Detection Settings", expanded=False):
        detect_peaks = st.checkbox("Detect Peaks", value=True)
        distance = st.slider("Min Distance Between Peaks (frames)", 1, 100, fps // 10) if detect_peaks else None
        prominence = st.slider("Peak Prominence", 0.01, 1.0, 0.1)
        min_bpm = st.slider("Minimum BPM", 100, 400, 300)
        max_bpm = st.slider("Maximum BPM", 200, 600, 600)

    with st.sidebar.expander("📈 Step 6: HR Over Time", expanded=False):
        window_sec = st.slider("Window Size (s)", 2, 30, 10)
        step_sec = st.slider("Step Size (s)", 1, 10, 1)

    filtered = signal.copy()
    if apply_band:
        filtered = apply_bandpass_filter(filtered, lowcut, highcut, fps, order)
    if apply_savgol:
        filtered = savgol_filter(filtered, window_length, polyorder)

    mean_val = np.mean(filtered)
    std_val = np.std(filtered)
    z_scores = (filtered - mean_val) / std_val

    peak_indices = find_peaks(filtered, distance=distance, prominence=prominence)[0] if detect_peaks else []
    zscore_peaks = find_peaks(z_scores, distance=distance, prominence=prominence)[0]

    show_tabs(signal, filtered, z_scores, time, zscore_peaks, peak_indices, fps, prominence, min_bpm, max_bpm, window_sec, step_sec)

# ------------------ TABS ------------------ #
def show_tabs(signal, filtered, z_scores, time, zscore_peaks, peak_indices, fps, prominence, min_bpm, max_bpm, window_sec, step_sec):
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Raw + Filtered", "📊 Analysis Summary", "🔬 FFT Spectrum", "🫀 HR Over Time"])

    with tab1:
        st.subheader("Raw Surface Area Signal")
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(x=time, y=signal, mode="lines", name="Raw Signal"))
        fig_raw.update_layout(xaxis_title="Time (s)", yaxis_title="Surface Area", height=400)
        st.plotly_chart(fig_raw, use_container_width=True)

        st.subheader("Filtered Signal with Z-score Peaks")
        fig_filtered = go.Figure()
        fig_filtered.add_trace(go.Scatter(x=time, y=z_scores, mode="lines", name="Z-scores"))
        fig_filtered.add_trace(go.Scatter(
            x=time[zscore_peaks], y=z_scores[zscore_peaks], mode="markers",
            name="Z-score Peaks", marker=dict(color="red", size=8)))
        fig_filtered.update_layout(xaxis_title="Time (s)", yaxis_title="Z-score", height=400)
        st.plotly_chart(fig_filtered, use_container_width=True)

    with tab2:
        st.subheader("Summary Table")
        duration_sec = len(signal) / fps
        bpm_from_peaks = compute_peak_bpm(z_scores, fs=fps, min_bpm=min_bpm, max_bpm=max_bpm, prominence=prominence)
        _, _, peak_freq, _ = find_fft_peak(z_scores, fps)
        fft_bpm = peak_freq * 60

        summary_data = {
            "Metric": ["Number of Frames", "Duration (s)", "Detected Peaks", "BPM (Peaks)", "Dominant Freq (Hz)", "BPM (FFT)"],
            "Value": [len(signal), round(duration_sec, 2), len(peak_indices), round(bpm_from_peaks, 2), round(peak_freq, 2), round(fft_bpm, 2)]
        }
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
        
        csv = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Summary as CSV",
            data=csv,
            file_name='heartbeat_summary.csv',
            mime='text/csv'
        )


    with tab3:
        st.subheader("FFT Spectrum with Dominant Frequency")
        xf, yf = compute_fft(z_scores, fps)
        dominant_freq = xf[np.argmax(yf)]
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(x=xf, y=yf, mode='lines', name='FFT Spectrum'))
        fig_fft.add_trace(go.Scatter(x=[dominant_freq, dominant_freq], y=[0, max(yf)],
                                     mode='lines', name=f'Dominant: {dominant_freq:.2f} Hz',
                                     line=dict(color='red', width=2, dash='dash')))
        fig_fft.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Amplitude", height=500)
        st.plotly_chart(fig_fft, use_container_width=True)

    with tab4:
        st.subheader("Heart Rate Over Time (Sliding Window)")
        centers, bpm_peaks, bpm_ffts = sliding_window_bpm(
            z_scores, time, fps, window_sec, step_sec,
            min_bpm=min_bpm, max_bpm=max_bpm, prominence=prominence
        )
        fig_bpm = go.Figure()
        fig_bpm.add_trace(go.Scatter(x=centers, y=bpm_peaks, mode="lines+markers", name="BPM from Peaks"))
        fig_bpm.add_trace(go.Scatter(x=centers, y=bpm_ffts, mode="lines+markers", name="BPM from FFT"))
        fig_bpm.update_layout(xaxis_title="Time (s)", yaxis_title="Heart Rate (BPM)", height=400)
        st.plotly_chart(fig_bpm, use_container_width=True)

if __name__ == "__main__":
    main()
