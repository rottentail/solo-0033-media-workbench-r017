"""Predictive reconstruction of explicitly marked short source damage.

The estimator works on the native 48 kHz stereo source and never borrows
editorial window geometry: given the trusted samples immediately on both sides
of one marked interval it predicts the missing samples per channel, keeping the
two channels independent so their phase and relative level survive (an
anti-phase pair is never summed into silence).

Voiced (periodic) material is continued by a harmonic model of the estimated
fractional pitch period, least-squares fitted to both trusted sides and
evaluated at fractional phase, so non-integer periods keep their timing and
the local timbre (harmonic balance) is preserved rather than flattened toward a
single cycle or a straight line. Aperiodic material uses cubic Hermite
interpolation of the boundary values and slopes, and digital silence stays
silent. Samples outside the marked interval are never moved, rewritten or
faded; callers apply occurrence gain afterwards.
"""
import numpy as np

CONTEXT = 512
# Plausible pitch lags at 48 kHz: ~100 Hz (480 samples) to 1.5 kHz (32).
# The upper lag leaves anchor samples inside the fixed 512-sample trusted
# context; lower fundamentals have no complete cycle to anchor and use the
# aperiodic bridge instead of a spurious sub-harmonic.
MIN_LAG = 32
MAX_LAG = 480
# Normalized squared difference below this score means a convincingly periodic
# context (equivalent to normalized correlation above ~0.975). The tight gate
# also rejects strong harmonics of a fundamental whose true period is longer
# than the trusted context can resolve.
PERIODIC_SCORE = 0.05
# A longer lag is only treated as the fundamental (rather than a
# sub-harmonic of a shorter, equally valid lag) when its score is at least
# this fraction better. Absolute near-zero scores are compared directly.
SUBHARMONIC_MARGIN = 0.25
# Lags scoring below this absolute value are accepted as valid pitch.
STRONG_SCORE = 0.02
# Below this per-sample mean energy the context is treated as digital silence.
SILENCE_ENERGY = 1e-18
# Harmonic model size relative to the estimated period.
HARMONIC_RATIO = 0.4
# Bounds on the harmonic model so its least-squares solve stays small and the
# 2*CONTEXT trusted rows always over-determine the 2H+1 columns.
MIN_HARMONICS = 3
MAX_HARMONICS = 24


def read(handle, first, count):
    handle.seek(first * 8)
    raw = handle.read(count * 8)
    if len(raw) != count * 8:
        raise ValueError('incomplete repair context')
    pcm = np.frombuffer(raw, dtype='<f4').reshape(-1, 2).astype(np.float64)
    if not np.isfinite(pcm).all():
        raise ValueError('nonfinite repair context')
    return pcm


def _lag_scores(side):
    """Normalized squared-difference pitch scores for one mono context."""
    n = len(side)
    upper = min(MAX_LAG, n - MIN_LAG)
    scores = np.empty(upper - MIN_LAG + 1)
    for k, lag in enumerate(range(MIN_LAG, upper + 1)):
        a = side[lag:]
        b = side[:-lag]
        denom = float(np.dot(a, a) + np.dot(b, b))
        scores[k] = (float(np.dot(a - b, a - b)) / denom) if denom > 0.0 else 2.0
    return scores


def _refine_lag(scores, lag):
    """Parabolic refinement of the score minimum to a fractional lag."""
    idx = lag - MIN_LAG
    if 0 < idx < len(scores) - 1:
        s0, s1, s2 = float(scores[idx - 1]), float(scores[idx]), float(scores[idx + 1])
        denom = s0 - 2 * s1 + s2
        if abs(denom) > 1e-12:
            shift = 0.5 * (s0 - s2) / denom
            if abs(shift) <= 1.0:
                lag = lag + shift
    return float(lag)


def _estimate_period(left, right):
    """Shared fractional pitch period across both channels, else None.

    Each channel is scored independently so anti-phase channels contribute
    instead of cancelling. The accepted lag is the shortest lag within a
    relative margin of the global best score: a longer lag that fits no better
    than a shorter one is a sub-harmonic of the same harmonic series, not the
    fundamental. Material whose best score misses the periodicity gate,
    including fundamentals too long to fit one cycle in the trusted context,
    is bridged aperiodically.
    """
    best_score, best_scores = 2.0, None
    for channel in range(2):
        for side in (left[:, channel], right[:, channel]):
            energy = float(np.dot(side, side))
            if energy <= SILENCE_ENERGY * len(side):
                continue
            scores = _lag_scores(side)
            idx = int(np.argmin(scores))
            score = float(scores[idx])
            if score < best_score:
                best_score, best_scores = score, scores
    if best_scores is None or best_score > PERIODIC_SCORE:
        return None
    # Among all periodic-quality lags, keep the shortest whose score is not
    # clearly worse than the global best. This rejects sub-harmonic lags of a
    # harmonic series while accepting a genuine long fundamental.
    threshold = min(PERIODIC_SCORE,
                    best_score + SUBHARMONIC_MARGIN * max(best_score, STRONG_SCORE))
    candidates = np.flatnonzero(best_scores <= threshold)
    chosen = MIN_LAG + int(candidates[0])
    return _refine_lag(best_scores, chosen)


def _harmonic_design(indices, omega, harmonics):
    """Least-squares design matrix: constant plus sin/cos per harmonic."""
    indices = np.asarray(indices, dtype=np.float64)
    angles = np.multiply.outer(indices, omega)
    parts = [np.ones((len(indices), 1))]
    for h in range(1, harmonics + 1):
        parts.append(np.sin(h * angles).reshape(len(indices), -1))
        parts.append(np.cos(h * angles).reshape(len(indices), -1))
    return np.concatenate(parts, axis=1)


def _periodic_channel(side_l, side_r, period, length, harmonics):
    """Harmonic continuation for one channel at a fractional pitch period.

    A fixed set of harmonics of the fitted fractional fundamental is solved
    against every trusted sample on both sides in one continuous,
    phase-consistent coordinate (left context at negative indices, right
    context at indices beyond the gap) and evaluated across the gap at
    fractional sample phase. The prediction is left unmodified in the gap so
    its natural periodic phase is preserved: at a high pitch a gap may span
    several cycles and its boundary values legitimately differ.
    """
    omega = 2 * np.pi / period
    left_idx = np.arange(-len(side_l), 0, dtype=np.float64)
    right_idx = np.arange(length, length + len(side_r), dtype=np.float64)
    idx = np.concatenate((left_idx, right_idx))
    values = np.concatenate((side_l, side_r))
    design = _harmonic_design(idx, omega, harmonics)
    coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
    gap_idx = np.arange(length, dtype=np.float64)
    base = _harmonic_design(gap_idx, omega, harmonics) @ coeffs
    if not np.isfinite(base).all():
        raise ValueError('nonfinite harmonic reconstruction')
    return base


def _slope(side, at_left):
    if at_left:
        return side[-1] - side[-2] if len(side) >= 2 else 0.0
    return side[1] - side[0] if len(side) >= 2 else 0.0


def _hermite_channel(side_l, side_r, length):
    """Cubic Hermite bridge for aperiodic, non-silent material."""
    p0, p1 = side_l[-1], side_r[0]
    m0, m1 = _slope(side_l, True), _slope(side_r, False)
    u = np.arange(length, dtype=np.float64) / max(1, length - 1)
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def restore(left, right, length):
    """Predict `length` missing stereo samples between trusted contexts."""
    if length < 1 or len(left) < 2 or len(right) < 2:
        raise ValueError('repair needs trusted context on both sides')
    period = _estimate_period(left, right)
    # Keep the harmonic design well conditioned: the trusted rows must resolve
    # the highest harmonic, so never use more than roughly an eighth as many
    # harmonic columns as fit rows (2 sin/cos columns each plus the constant).
    fit_cap = (len(left) + len(right)) // 8
    harmonics = min(MAX_HARMONICS, fit_cap,
                    max(MIN_HARMONICS, int(period * HARMONIC_RATIO))) if period is not None \
        else MIN_HARMONICS
    repaired = np.empty((length, 2), dtype=np.float64)
    for channel in range(2):
        side_l, side_r = left[:, channel], right[:, channel]
        left_energy = float(np.dot(side_l, side_l))
        right_energy = float(np.dot(side_r, side_r))
        if max(left_energy, right_energy) < SILENCE_ENERGY * len(side_l):
            repaired[:, channel] = 0.0
        elif period is not None:
            repaired[:, channel] = _periodic_channel(side_l, side_r, period, length, harmonics)
        else:
            repaired[:, channel] = _hermite_channel(side_l, side_r, length)
    if not np.isfinite(repaired).all():
        raise ValueError('nonfinite repair reconstruction')
    return repaired.astype('<f4')


def patches(handle, intervals):
    """Reconstruct marked source intervals using full native context."""
    restored = []
    for first, stop in intervals:
        left = read(handle, first - CONTEXT, CONTEXT)
        right = read(handle, stop, CONTEXT)
        restored.append((first, stop, restore(left, right, stop - first)))
    return restored


def blocks(handle, clip, restored, first, stop, block_frames):
    """Yield repaired, gain-scaled PCM blocks for one occurrence view."""
    origin = clip['start_ms'] * 48
    for cursor in range(first, stop, block_frames):
        end = min(stop, cursor + block_frames)
        pcm = read(handle, origin + cursor, end - cursor)
        for left, right, patch in restored:
            a, b = max(origin + cursor, left), min(origin + end, right)
            if a < b:
                pcm[a - origin - cursor:b - origin - cursor] = patch[a - left:b - left]
        yield (pcm * clip['gain']).astype('<f4')
