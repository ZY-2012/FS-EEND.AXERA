#include "feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

extern "C" {
#include "kiss_fft.h"
}

namespace {
const int FFT_SIZE = 1024;
const int FRAME_SIZE = 200;   // win_length
const int HOP = 80;
const int MEL_BINS = 23;
const int CONTEXT = 7;
const int SUBSAMPLE = 10;
const int N_FREQS = FFT_SIZE / 2 + 1;  // 513
const double PI = 3.14159265358979323846;

// Slaney mel scale, matching librosa.filters.mel(..., htk=False) which is the
// default. Linear below 1 kHz, logarithmic above.
const double kFsp = 200.0 / 3.0;
const double kMinLogHz = 1000.0;
const double kMinLogMel = kMinLogHz / kFsp;  // 15.0
const double kLogStep = std::log(6.4) / 27.0;

double hz_to_mel(double hz) {
    if (hz < kMinLogHz) return hz / kFsp;
    return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
}

double mel_to_hz(double mel) {
    if (mel < kMinLogMel) return kFsp * mel;
    return kMinLogHz * std::exp(kLogStep * (mel - kMinLogMel));
}

// Reproduces librosa.filters.mel(sr=8000, n_fft=1024, n_mels=23) with the
// default norm='slaney': triangular filters over continuous FFT frequencies,
// each scaled by 2/(f[i+2]-f[i]).
std::vector<double> build_mel_filterbank(double sample_rate) {
    std::vector<double> mel_hz(MEL_BINS + 2);
    const double mel_min = hz_to_mel(0.0);
    const double mel_max = hz_to_mel(sample_rate / 2.0);
    for (int i = 0; i < MEL_BINS + 2; ++i) {
        const double mel = mel_min + (mel_max - mel_min) * i / (MEL_BINS + 1);
        mel_hz[i] = mel_to_hz(mel);
    }

    std::vector<double> fft_freqs(N_FREQS);
    for (int k = 0; k < N_FREQS; ++k)
        fft_freqs[k] = (sample_rate / 2.0) * k / (N_FREQS - 1);

    std::vector<double> weights(static_cast<size_t>(MEL_BINS) * N_FREQS, 0.0);
    for (int m = 0; m < MEL_BINS; ++m) {
        const double f_lo = mel_hz[m], f_ctr = mel_hz[m + 1], f_hi = mel_hz[m + 2];
        const double lo_diff = f_ctr - f_lo, hi_diff = f_hi - f_ctr;
        const double enorm = 2.0 / (f_hi - f_lo);  // norm='slaney'
        for (int k = 0; k < N_FREQS; ++k) {
            const double lower = (fft_freqs[k] - f_lo) / lo_diff;
            const double upper = (f_hi - fft_freqs[k]) / hi_diff;
            const double value = std::min(lower, upper);
            if (value > 0.0)
                weights[static_cast<size_t>(m) * N_FREQS + k] = value * enorm;
        }
    }
    return weights;
}
}  // namespace

std::vector<float> extract_ls_eend_features(const AudioData& a) {
    if (a.sample_rate != 8000) throw std::runtime_error("feature input must be 8 kHz");

    // librosa.stft(center=True, pad_mode='constant') puts frame t centered at
    // sample t*HOP and yields 1 + len(y)//HOP frames. The 200-sample window sits
    // in the middle of the 1024-point FFT buffer; samples outside the signal are
    // zero. datasets/feature.py then drops the last frame when the sample count
    // is a multiple of HOP.
    int n_frames = 1 + static_cast<int>(a.samples.size() / HOP);
    if (a.samples.size() % HOP == 0 && n_frames > 0) --n_frames;

    const std::vector<double> mel_fb = build_mel_filterbank(8000.0);

    // Periodic Hann, i.e. get_window('hann', 200, fftbins=True): the denominator
    // is FRAME_SIZE, not FRAME_SIZE-1.
    std::vector<double> window(FRAME_SIZE);
    for (int n = 0; n < FRAME_SIZE; ++n)
        window[n] = 0.5 - 0.5 * std::cos(2.0 * PI * n / FRAME_SIZE);

    kiss_fft_cfg cfg = kiss_fft_alloc(FFT_SIZE, 0, NULL, NULL);
    if (!cfg) throw std::runtime_error("KissFFT allocation failed");

    std::vector<float> mel(static_cast<size_t>(n_frames) * MEL_BINS, 0.0f);
    std::vector<kiss_fft_cpx> input(FFT_SIZE), spectrum(FFT_SIZE);
    std::vector<double> power(N_FREQS);
    const int window_offset = (FFT_SIZE - FRAME_SIZE) / 2;  // 412
    const int half_window = FRAME_SIZE / 2;                 // 100

    for (int t = 0; t < n_frames; ++t) {
        std::fill(input.begin(), input.end(), kiss_fft_cpx{0.0f, 0.0f});
        const int center = t * HOP;
        for (int n = 0; n < FRAME_SIZE; ++n) {
            const int idx = center - half_window + n;
            if (idx < 0 || idx >= static_cast<int>(a.samples.size())) continue;
            input[window_offset + n].r =
                static_cast<float>(a.samples[static_cast<size_t>(idx)] * window[n]);
        }

        kiss_fft(cfg, input.data(), spectrum.data());
        for (int k = 0; k < N_FREQS; ++k) {
            const double re = spectrum[k].r, im = spectrum[k].i;
            power[k] = re * re + im * im;  // |stft|^2
        }
        for (int m = 0; m < MEL_BINS; ++m) {
            const double* row = &mel_fb[static_cast<size_t>(m) * N_FREQS];
            double energy = 0.0;
            for (int k = 0; k < N_FREQS; ++k) energy += power[k] * row[k];
            mel[static_cast<size_t>(t) * MEL_BINS + m] =
                static_cast<float>(std::log10(std::max(energy, 1e-10)));
        }
    }
    kiss_fft_free(cfg);

    // feat_type=logmel23_cummn: subtract the cumulative mean over frames 0..t.
    std::vector<double> cumulative(MEL_BINS, 0.0);
    for (int t = 0; t < n_frames; ++t) {
        for (int m = 0; m < MEL_BINS; ++m) {
            const size_t at = static_cast<size_t>(t) * MEL_BINS + m;
            cumulative[m] += mel[at];
            mel[at] -= static_cast<float>(cumulative[m] / (t + 1));
        }
    }

    // +-CONTEXT frame splice with zero padding, then keep every SUBSAMPLE-th frame.
    std::vector<float> features;
    features.reserve(static_cast<size_t>((n_frames + SUBSAMPLE - 1) / SUBSAMPLE) *
                     (2 * CONTEXT + 1) * MEL_BINS);
    for (int t = 0; t < n_frames; t += SUBSAMPLE) {
        for (int d = -CONTEXT; d <= CONTEXT; ++d) {
            const int q = t + d;
            if (q < 0 || q >= n_frames) {
                features.insert(features.end(), MEL_BINS, 0.0f);
            } else {
                features.insert(features.end(),
                                mel.begin() + static_cast<size_t>(q) * MEL_BINS,
                                mel.begin() + static_cast<size_t>(q + 1) * MEL_BINS);
            }
        }
    }
    return features;
}
