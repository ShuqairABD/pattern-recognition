"""
Financial Pattern Recognition System
=====================================
Implements all 10 mathematical modules from the specification:
1.  Input data          - discrete price series P(t_i)
2.  Smoothing           - EMA / Savitzky-Golay / Gaussian
3.  Extrema detection   - discrete derivative sign changes
4.  Wave definition     - amplitude, duration, angle, velocity
5.  Classification      - impulse vs correction
6.  Structural triple   - W1→W2→W3 base pattern
7.  Correction ratio    - R = A2/A1  ∈ [0.3, 0.8]
8.  Phase model         - phase 1/2/3
9.  Fractality          - self-similarity across timeframes
10. Quality metric      - S = w1·f_ratio + w2·f_symmetry + w3·f_slope
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Extremum:
    index: int
    time: float
    price: float
    kind: str          # 'max' | 'min'


@dataclass
class Wave:
    idx: int
    start: Extremum
    end: Extremum
    amplitude: float   # A_j = |P(e_{j+1}) - P(e_j)|
    duration: float    # T_j = t_{j+1} - t_j
    angle: float       # θ_j = arctan(A_j / T_j)
    velocity: float    # V_j = A_j / T_j
    direction: str     # 'up' | 'down'
    wave_type: str = "unknown"   # 'impulse' | 'correction'


@dataclass
class StructuralTriple:
    w1: Wave
    w2: Wave
    w3: Wave
    correction_ratio: float   # R = A2 / A1
    quality_score: float      # S = w1·f_ratio + w2·f_symmetry + w3·f_slope
    phase: int                # 1=impulse, 2=correction, 3=continuation
    is_valid: bool


# ─────────────────────────────────────────────
# Module 1 & 2  –  Data + Smoothing
# ─────────────────────────────────────────────

class DataProcessor:
    """
    Module 1: P(t_i), i = 1..N
    Module 2: P̃(t) after noise filtering
    """

    def __init__(self, method: str = "savgol", window: int = 11, poly: int = 3):
        self.method = method
        self.window = window
        self.poly   = poly

    def load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["time"])
        df = df.sort_values("time").reset_index(drop=True)
        return df

    def load_series(self, prices: np.ndarray,
                    times: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Accept a raw numpy array (for quick testing / API integration)."""
        if times is None:
            times = np.arange(len(prices), dtype=float)
        return pd.DataFrame({"time": times, "close": prices})

    def smooth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Module 2: P̃(t)
        Variants: 'ema' | 'savgol' | 'gaussian'
        """
        p = df["close"].values.astype(float)
        w = max(self.window, 5)

        if self.method == "ema":
            alpha = 2 / (w + 1)
            s = pd.Series(p).ewm(span=w, adjust=False).mean().values
        elif self.method == "gaussian":
            sigma = w / 4
            s = gaussian_filter1d(p, sigma=sigma)
        else:  # savgol (default)
            wl = w if w % 2 == 1 else w + 1
            wl = max(wl, self.poly + 2)
            s = savgol_filter(p, window_length=wl, polyorder=self.poly)

        df = df.copy()
        df["smooth"] = s
        return df


# ─────────────────────────────────────────────
# Module 3  –  Extrema Detection
# ─────────────────────────────────────────────

class ExtremaDetector:
    """
    Module 3:
    ΔP_i = P̃(t_i) − P̃(t_{i−1})
    max  if ΔP_i > 0 and ΔP_{i+1} < 0
    min  if ΔP_i < 0 and ΔP_{i+1} > 0
    → E = {e_1, e_2, ..., e_k}
    """

    def __init__(self, min_distance: int = 3):
        self.min_distance = min_distance

    def detect(self, df: pd.DataFrame) -> List[Extremum]:
        col   = "smooth" if "smooth" in df.columns else "close"
        p     = df[col].values
        times = df["time"].values

        # Discrete derivative
        delta = np.diff(p, prepend=p[0])          # ΔP_i

        extrema: List[Extremum] = []
        last_idx = -self.min_distance

        for i in range(1, len(p) - 1):
            if (i - last_idx) < self.min_distance:
                continue
            if delta[i] > 0 and delta[i + 1] < 0:
                extrema.append(Extremum(i, float(i), float(p[i]), "max"))
                last_idx = i
            elif delta[i] < 0 and delta[i + 1] > 0:
                extrema.append(Extremum(i, float(i), float(p[i]), "min"))
                last_idx = i

        # Remove duplicates of same kind that are adjacent
        extrema = self._filter_alternating(extrema, p)
        return extrema

    def _filter_alternating(self, extrema: List[Extremum],
                            p: np.ndarray) -> List[Extremum]:
        """Ensure strict alternation max–min–max–..."""
        if not extrema:
            return extrema
        filtered = [extrema[0]]
        for e in extrema[1:]:
            if e.kind == filtered[-1].kind:
                # Keep the more extreme one
                if e.kind == "max" and e.price > filtered[-1].price:
                    filtered[-1] = e
                elif e.kind == "min" and e.price < filtered[-1].price:
                    filtered[-1] = e
            else:
                filtered.append(e)
        return filtered


# ─────────────────────────────────────────────
# Module 4  –  Wave Builder
# ─────────────────────────────────────────────

class WaveBuilder:
    """
    Module 4:
    W_j = (e_j, e_{j+1})
    A_j = |P(e_{j+1}) − P(e_j)|
    T_j = t_{j+1} − t_j
    θ_j = arctan(A_j / T_j)
    V_j = A_j / T_j
    """

    def build(self, extrema: List[Extremum]) -> List[Wave]:
        waves = []
        for j in range(len(extrema) - 1):
            s = extrema[j]
            e = extrema[j + 1]
            A = abs(e.price - s.price)
            T = abs(e.time  - s.time) or 1e-9   # avoid division by zero
            theta = np.arctan(A / T)
            V = A / T
            direction = "up" if e.price > s.price else "down"
            waves.append(Wave(
                idx=j, start=s, end=e,
                amplitude=A, duration=T,
                angle=theta, velocity=V,
                direction=direction
            ))
        return waves


# ─────────────────────────────────────────────
# Module 5  –  Impulse / Correction Classifier
# ─────────────────────────────────────────────

class WaveClassifier:
    """
    Module 5:
    Ā  = (1/M) Σ A_j
    Impulse:    A_j > λ1·Ā  and  V_j > λ2·V̄
    Correction: A_j < λ1·Ā
    """

    def __init__(self, lambda1: float = 0.85, lambda2: float = 0.70):
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def classify(self, waves: List[Wave]) -> List[Wave]:
        if not waves:
            return waves

        A_mean = np.mean([w.amplitude for w in waves])
        V_mean = np.mean([w.velocity  for w in waves])

        for w in waves:
            if (w.amplitude > self.lambda1 * A_mean and
                    w.velocity > self.lambda2 * V_mean):
                w.wave_type = "impulse"
            else:
                w.wave_type = "correction"
        return waves


# ─────────────────────────────────────────────
# Module 6 + 7  –  Structural Triple
# ─────────────────────────────────────────────

class StructuralTripleDetector:
    """
    Module 6: W1→W2→W3  where W1=impulse, W2=correction, W3=impulse, A3≥α·A1
    Module 7: R = A2/A1  ∈ [0.3, 0.8]
    """

    def __init__(self, alpha: float = 0.618,
                 r_min: float = 0.3, r_max: float = 0.8):
        self.alpha = alpha
        self.r_min = r_min
        self.r_max = r_max

    def find_triples(self, waves: List[Wave]) -> List[StructuralTriple]:
        triples = []
        for i in range(len(waves) - 2):
            w1, w2, w3 = waves[i], waves[i+1], waves[i+2]
            R     = w2.amplitude / (w1.amplitude + 1e-9)
            valid = (
                w1.wave_type == "impulse"     and
                w2.wave_type == "correction"  and
                w3.wave_type == "impulse"     and
                w3.amplitude >= self.alpha * w1.amplitude and
                self.r_min <= R <= self.r_max
            )
            # Opposite directions: W1 and W2 must differ
            if w1.direction == w2.direction:
                valid = False

            triples.append(StructuralTriple(
                w1=w1, w2=w2, w3=w3,
                correction_ratio=R,
                quality_score=0.0,   # filled by QualityMetric
                phase=0,             # filled by PhaseModel
                is_valid=valid
            ))
        return triples


# ─────────────────────────────────────────────
# Module 8  –  Phase Model
# ─────────────────────────────────────────────

class PhaseModel:
    """
    Module 8:
    Phase 1 = first impulse  (inside W1)
    Phase 2 = correction     (inside W2)
    Phase 3 = continuation   (inside W3 or beyond)
    """

    def assign(self, triples: List[StructuralTriple],
               current_idx: int) -> Tuple[int, str]:
        """
        Returns (phase_number, description) for current_idx position
        relative to the last valid triple.
        """
        valid = [t for t in triples if t.is_valid]
        if not valid:
            return 0, "No valid structure found"

        last = valid[-1]
        w1_end = int(last.w1.end.index)
        w2_end = int(last.w2.end.index)
        w3_end = int(last.w3.end.index)

        if current_idx <= w1_end:
            phase, desc = 1, "Phase 1 — First Impulse 🚀"
        elif current_idx <= w2_end:
            phase, desc = 2, "Phase 2 — Correction 🔄"
        elif current_idx <= w3_end:
            phase, desc = 3, "Phase 3 — Continuation Impulse 📈"
        else:
            phase, desc = 3, "Phase 3+ — Post-structure zone 🔍"

        for t in valid:
            t.phase = phase
        return phase, desc


# ─────────────────────────────────────────────
# Module 9  –  Fractality
# ─────────────────────────────────────────────

class FractalityAnalyzer:
    """
    Module 9:
    W_j^(TF1) ⊂ W_k^(TF2)  →  A_j^(TF1) / A_k^(TF2) ≈ const
    Self-similarity coefficient
    """

    def self_similarity(self,
                        waves_tf1: List[Wave],
                        waves_tf2: List[Wave]) -> Dict:
        if not waves_tf1 or not waves_tf2:
            return {"coefficient": None, "stable": False}

        ratios = []
        for w1 in waves_tf1:
            for w2 in waves_tf2:
                if w2.amplitude > 1e-9:
                    ratios.append(w1.amplitude / w2.amplitude)

        if not ratios:
            return {"coefficient": None, "stable": False}

        coeff = float(np.median(ratios))
        cv    = float(np.std(ratios) / (np.mean(ratios) + 1e-9))  # CV
        return {
            "coefficient": coeff,
            "cv":          cv,
            "stable":      cv < 0.3,   # stable if CV < 30%
            "n_pairs":     len(ratios)
        }


# ─────────────────────────────────────────────
# Module 10  –  Quality Metric
# ─────────────────────────────────────────────

class QualityMetric:
    """
    Module 10:
    S = w1·f_ratio + w2·f_symmetry + w3·f_slope
    f_ratio     – proportion conformity
    f_symmetry  – temporal symmetry
    f_slope     – direction stability
    S > threshold → structure accepted
    """

    def __init__(self, w1: float = 0.4, w2: float = 0.3, w3: float = 0.3,
                 threshold: float = 0.55):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.threshold = threshold

    def _f_ratio(self, triple: StructuralTriple) -> float:
        """Correction ratio closeness to golden zone [0.3, 0.8]."""
        R = triple.correction_ratio
        if 0.3 <= R <= 0.8:
            # Peak at 0.618 (Fibonacci)
            return 1.0 - abs(R - 0.618) / 0.318
        return max(0.0, 1.0 - abs(R - 0.55) / 0.55)

    def _f_symmetry(self, triple: StructuralTriple) -> float:
        """Temporal symmetry: T1 ≈ T3."""
        t1, t3 = triple.w1.duration, triple.w3.duration
        denom = max(t1, t3) + 1e-9
        return 1.0 - abs(t1 - t3) / denom

    def _f_slope(self, triple: StructuralTriple) -> float:
        """W1 and W3 should have same direction; W2 opposite."""
        d1, d3 = triple.w1.direction, triple.w3.direction
        d2     = triple.w2.direction
        same_impulse  = 1.0 if d1 == d3        else 0.0
        diff_correct  = 1.0 if d2 != d1        else 0.0
        return (same_impulse + diff_correct) / 2.0

    def score(self, triples: List[StructuralTriple]) -> List[StructuralTriple]:
        for t in triples:
            fr   = self._f_ratio(t)
            fs   = self._f_symmetry(t)
            fsl  = self._f_slope(t)
            t.quality_score = self.w1 * fr + self.w2 * fs + self.w3 * fsl
            if t.quality_score < self.threshold:
                t.is_valid = False
        return triples


# ─────────────────────────────────────────────
# Visualizer
# ─────────────────────────────────────────────

class Visualizer:

    PHASE_COLORS = {1: "#00B4D8", 2: "#FFB703", 3: "#06D6A0"}
    WAVE_COLORS  = {"impulse": "#06D6A0", "correction": "#FFB703",
                    "unknown": "#ADB5BD"}

    def plot(self, df: pd.DataFrame,
             extrema: List[Extremum],
             waves: List[Wave],
             triples: List[StructuralTriple],
             current_phase: int,
             phase_desc: str,
             title: str = "Pattern Recognition") -> go.Figure:

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.75, 0.25],
            shared_xaxes=True,
            subplot_titles=(title, "Wave Amplitude Profile")
        )

        x = list(range(len(df)))

        # ── Raw price ──────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=x, y=df["close"].values,
            mode="lines", name="Price",
            line=dict(color="#4A4E69", width=1.5),
            opacity=0.6
        ), row=1, col=1)

        # ── Smoothed price ────────────────────────────────────
        if "smooth" in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df["smooth"].values,
                mode="lines", name="Smoothed P̃(t)",
                line=dict(color="#9A8C98", width=2, dash="dot"),
            ), row=1, col=1)

        # ── Extrema ───────────────────────────────────────────
        max_e = [e for e in extrema if e.kind == "max"]
        min_e = [e for e in extrema if e.kind == "min"]

        fig.add_trace(go.Scatter(
            x=[e.index for e in max_e],
            y=[e.price  for e in max_e],
            mode="markers", name="Maxima",
            marker=dict(symbol="triangle-up", size=10, color="#E63946")
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[e.index for e in min_e],
            y=[e.price  for e in min_e],
            mode="markers", name="Minima",
            marker=dict(symbol="triangle-down", size=10, color="#2A9D8F")
        ), row=1, col=1)

        # ── Waves (colored) ────────────────────────────────────
        for w in waves:
            color = self.WAVE_COLORS.get(w.wave_type, "#ADB5BD")
            fig.add_trace(go.Scatter(
                x=[w.start.index, w.end.index],
                y=[w.start.price, w.end.price],
                mode="lines+markers", showlegend=False,
                line=dict(color=color, width=2.5),
                marker=dict(size=5)
            ), row=1, col=1)

            # Label
            mid_x = (w.start.index + w.end.index) / 2
            mid_y = (w.start.price + w.end.price) / 2
            label = "I" if w.wave_type == "impulse" else "C"
            fig.add_annotation(
                x=mid_x, y=mid_y,
                text=f"<b>{label}</b> A={w.amplitude:.2f}",
                showarrow=False,
                font=dict(size=9, color=color),
                row=1, col=1
            )

        # ── Valid triples highlighted ──────────────────────────
        valid = [t for t in triples if t.is_valid]
        for t in valid:
            x0 = t.w1.start.index
            x1 = t.w3.end.index
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=self.PHASE_COLORS.get(t.phase, "#CCC"),
                opacity=0.08, line_width=0, row=1, col=1
            )
            fig.add_annotation(
                x=(x0 + x1) / 2,
                y=df["close"].max() * 1.001,
                text=f"<b>S={t.quality_score:.2f}</b>",
                showarrow=False,
                font=dict(size=9, color="#555"),
                row=1, col=1
            )

        # ── Wave amplitude bar chart ───────────────────────────
        bar_colors = [self.WAVE_COLORS.get(w.wave_type, "#ADB5BD") for w in waves]
        fig.add_trace(go.Bar(
            x=[w.idx for w in waves],
            y=[w.amplitude for w in waves],
            name="Amplitude A_j",
            marker_color=bar_colors, opacity=0.8
        ), row=2, col=1)

        # ── Phase annotation ──────────────────────────────────
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            text=f"<b>Current: {phase_desc}</b>",
            showarrow=False, align="left",
            bgcolor=self.PHASE_COLORS.get(current_phase, "#DDD"),
            bordercolor="#333", borderwidth=1,
            font=dict(size=13, color="#111")
        )

        fig.update_layout(
            template="plotly_white",
            height=700,
            legend=dict(orientation="h", y=-0.05),
            margin=dict(l=50, r=50, t=60, b=40),
        )
        return fig


# ─────────────────────────────────────────────
# PatternSystem  –  Main Orchestrator
# ─────────────────────────────────────────────

class PatternSystem:
    """
    Full pipeline: data → smooth → extrema → waves →
    classify → triples → quality → phase → visualize
    """

    def __init__(
        self,
        smooth_method:   str   = "savgol",
        smooth_window:   int   = 11,
        smooth_poly:     int   = 3,
        min_ext_dist:    int   = 3,
        lambda1:         float = 0.85,
        lambda2:         float = 0.70,
        alpha:           float = 0.618,
        r_min:           float = 0.30,
        r_max:           float = 0.80,
        quality_w1:      float = 0.40,
        quality_w2:      float = 0.30,
        quality_w3:      float = 0.30,
        quality_thresh:  float = 0.55,
    ):
        self.processor  = DataProcessor(smooth_method, smooth_window, smooth_poly)
        self.detector   = ExtremaDetector(min_ext_dist)
        self.builder    = WaveBuilder()
        self.classifier = WaveClassifier(lambda1, lambda2)
        self.tripler    = StructuralTripleDetector(alpha, r_min, r_max)
        self.phaser     = PhaseModel()
        self.metric     = QualityMetric(quality_w1, quality_w2,
                                        quality_w3, quality_thresh)
        self.fractal    = FractalityAnalyzer()
        self.visualizer = Visualizer()

    # ── Core run ──────────────────────────────────────────────

    def run(self, prices: np.ndarray,
            times: Optional[np.ndarray] = None,
            title: str = "Pattern Recognition") -> Dict:
        """
        prices: 1-D numpy array of close prices
        times:  optional index array (defaults to 0..N-1)
        Returns: dict with all results + plotly figure
        """
        # 1 + 2  Data & Smoothing
        df = self.processor.load_series(prices, times)
        df = self.processor.smooth(df)

        # 3  Extrema  E = {e_1,...,e_k}
        extrema = self.detector.detect(df)

        # 4  Waves  W_j = (e_j, e_{j+1})
        waves = self.builder.build(extrema)

        # 5  Classify impulse / correction
        waves = self.classifier.classify(waves)

        # 6 + 7  Structural triples + correction ratio
        triples = self.tripler.find_triples(waves)

        # 10  Quality metric  S = w1·f_ratio + w2·f_symmetry + w3·f_slope
        triples = self.metric.score(triples)

        # 8  Phase model
        current_phase, phase_desc = self.phaser.assign(triples, len(df) - 1)

        # Build summary report
        valid_triples = [t for t in triples if t.is_valid]
        report = self._build_report(waves, valid_triples, current_phase,
                                    phase_desc, len(df))

        # Visualize
        fig = self.visualizer.plot(
            df, extrema, waves, triples,
            current_phase, phase_desc, title
        )

        return {
            "dataframe":      df,
            "extrema":        extrema,
            "waves":          waves,
            "triples":        triples,
            "valid_triples":  valid_triples,
            "current_phase":  current_phase,
            "phase_desc":     phase_desc,
            "report":         report,
            "figure":         fig,
        }

    def run_csv(self, path: str, title: str = "Pattern Recognition") -> Dict:
        df_raw = self.processor.load_csv(path)
        return self.run(df_raw["close"].values, title=title)

    def run_multiframe(self, data: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        data = {"1m": prices_1m, "5m": prices_5m, "1h": prices_1h, ...}
        Returns results per timeframe + fractal similarity report.
        """
        results = {}
        for tf, prices in data.items():
            results[tf] = self.run(prices, title=f"TF: {tf}")

        # Module 9 – fractality between consecutive timeframes
        tfs  = list(results.keys())
        frac = {}
        for i in range(len(tfs) - 1):
            tf1, tf2 = tfs[i], tfs[i + 1]
            w1 = results[tf1]["waves"]
            w2 = results[tf2]["waves"]
            frac[f"{tf1}→{tf2}"] = self.fractal.self_similarity(w1, w2)

        return {"timeframes": results, "fractality": frac}

    # ── Report ────────────────────────────────────────────────

    def _build_report(self, waves, valid_triples,
                      phase, phase_desc, n) -> str:
        impulses   = [w for w in waves if w.wave_type == "impulse"]
        corrections= [w for w in waves if w.wave_type == "correction"]
        lines = [
            "=" * 50,
            "  PATTERN RECOGNITION REPORT",
            "=" * 50,
            f"  Total bars analysed  : {n}",
            f"  Waves detected       : {len(waves)}",
            f"    Impulses           : {len(impulses)}",
            f"    Corrections        : {len(corrections)}",
            f"  Valid triples (W1W2W3): {len(valid_triples)}",
            "-" * 50,
        ]
        if valid_triples:
            lines.append("  VALID STRUCTURES:")
            for i, t in enumerate(valid_triples, 1):
                lines += [
                    f"  [{i}] A1={t.w1.amplitude:.4f}  "
                    f"A2={t.w2.amplitude:.4f}  A3={t.w3.amplitude:.4f}",
                    f"      R={t.correction_ratio:.3f}  "
                    f"Quality S={t.quality_score:.3f}  "
                    f"Phase={t.phase}",
                ]
        lines += [
            "-" * 50,
            f"  CURRENT PHASE  : {phase_desc}",
            "=" * 50,
        ]
        return "\n".join(lines)
