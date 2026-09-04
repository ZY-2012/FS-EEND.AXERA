#include "streaming_diarizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

std::string enc_name(int i, const char* suffix) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "enc%d_%s", i, suffix);
    return buf;
}

std::string dec_name(int i, const char* suffix) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "dec%d_%s", i, suffix);
    return buf;
}

// mean += (inc - mean) * b, in place.
void accumulate(std::vector<float>& mean, const float* inc, float b) {
    const float keep = 1.0f - b;
    for (size_t i = 0; i < mean.size(); ++i) mean[i] = mean[i] * keep + inc[i] * b;
}

// Median filter of odd length over a binary column; matches scipy.signal.medfilt
// with zero padding at the edges, i.e. majority vote over the window.
std::vector<int> median_filter(const std::vector<int>& column, int median) {
    if (median <= 1) return column;
    const int half = median / 2;
    const int n = static_cast<int>(column.size());
    std::vector<int> out(column.size(), 0);
    for (int t = 0; t < n; ++t) {
        int ones = 0;
        for (int k = -half; k <= half; ++k) {
            const int idx = t + k;
            if (idx >= 0 && idx < n) ones += column[idx];
        }
        out[t] = (ones > median / 2) ? 1 : 0;
    }
    return out;
}

}  // namespace

StreamingDiarizer::StreamingDiarizer(const std::string& model_path) : runner_(model_path) {
    if (!runner_.has_input("feat") || !runner_.has_input("inv_count") ||
        !runner_.has_input("dec_inv_count") || !runner_.has_input("conv_cache")) {
        throw std::runtime_error(
            "model does not look like the LS-EEND streaming graph "
            "(expected feat / inv_count / dec_inv_count / conv_cache inputs)");
    }
    slots_ = static_cast<int>(runner_.output_elems("pred"));
    reset();
}

void StreamingDiarizer::reset() {
    enc_mean_.assign(kEncLayers, {});
    enc_conv_.assign(kEncLayers, {});
    dec_mean_.assign(kDecLayers, {});
    for (int i = 0; i < kEncLayers; ++i) {
        enc_mean_[i].assign(runner_.input_elems(enc_name(i, "kv")), 0.0f);
        enc_conv_[i].assign(runner_.input_elems(enc_name(i, "conv")), 0.0f);
    }
    for (int i = 0; i < kDecLayers; ++i)
        dec_mean_[i].assign(runner_.input_elems(dec_name(i, "kv")), 0.0f);
    conv_cache_.assign(runner_.input_elems("conv_cache"), 0.0f);
    enc_t_ = 0;
    dec_t_ = 0;
}

DiarResult StreamingDiarizer::run(const std::vector<float>& features, int frames,
                                  double duration_sec, int max_speakers, float threshold,
                                  int median) {
    const size_t feat_dim = runner_.input_elems("feat");
    if (features.size() < static_cast<size_t>(frames) * feat_dim)
        throw std::runtime_error("feature buffer smaller than frames * feat_dim");
    if (frames <= kConvDelay)
        throw std::runtime_error("recording too short: need more than 9 feature frames");

    reset();
    DiarResult result;
    result.frames = frames;
    result.duration_sec = duration_sec;
    result.slots = slots_;
    result.logits.reserve(static_cast<size_t>(frames - kConvDelay) * slots_);

    const size_t inv_elems = runner_.input_elems("inv_count");
    const size_t dec_inv_elems = runner_.input_elems("dec_inv_count");

    const auto t_start = std::chrono::steady_clock::now();
    for (int t = 0; t < frames; ++t) {
        const float enc_b = 1.0f / static_cast<float>(enc_t_ + 1);
        const float dec_b = 1.0f / static_cast<float>(std::max(dec_t_ + 1, 1));

        std::memcpy(runner_.input("feat"), features.data() + static_cast<size_t>(t) * feat_dim,
                    feat_dim * sizeof(float));
        std::fill_n(runner_.input("inv_count"), inv_elems, enc_b);
        std::fill_n(runner_.input("dec_inv_count"), dec_inv_elems, dec_b);
        for (int i = 0; i < kEncLayers; ++i) {
            std::memcpy(runner_.input(enc_name(i, "kv")), enc_mean_[i].data(),
                        enc_mean_[i].size() * sizeof(float));
            std::memcpy(runner_.input(enc_name(i, "conv")), enc_conv_[i].data(),
                        enc_conv_[i].size() * sizeof(float));
        }
        std::memcpy(runner_.input("conv_cache"), conv_cache_.data(),
                    conv_cache_.size() * sizeof(float));
        for (int i = 0; i < kDecLayers; ++i)
            std::memcpy(runner_.input(dec_name(i, "kv")), dec_mean_[i].data(),
                        dec_mean_[i].size() * sizeof(float));

        runner_.run();

        // Encoder state always advances.
        for (int i = 0; i < kEncLayers; ++i) {
            accumulate(enc_mean_[i], runner_.output(enc_name(i, "inc")), enc_b);
            const float* conv_out = runner_.output(enc_name(i, "conv_out"));
            std::copy(conv_out, conv_out + enc_conv_[i].size(), enc_conv_[i].begin());
        }
        const float* cache_out = runner_.output("conv_cache_out");
        std::copy(cache_out, cache_out + conv_cache_.size(), conv_cache_.begin());
        ++enc_t_;

        // The output conv emits nothing for the first kConvDelay calls, so the
        // native loop never invokes the decoder there: drop pred, freeze state.
        if (enc_t_ <= kConvDelay) continue;

        for (int i = 0; i < kDecLayers; ++i)
            accumulate(dec_mean_[i], runner_.output(dec_name(i, "inc")), dec_b);
        ++dec_t_;

        const float* pred = runner_.output("pred");
        result.logits.insert(result.logits.end(), pred, pred + slots_);
    }
    const auto t_end = std::chrono::steady_clock::now();

    result.emitted_frames = static_cast<int>(result.logits.size() / slots_);
    result.latency_ms_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.latency_ms_per_frame = result.latency_ms_total / std::max(frames, 1);
    if (duration_sec > 0.0) result.rtf = (result.latency_ms_total / 1000.0) / duration_sec;

    result.segments = decode_segments(result.logits, result.emitted_frames, slots_,
                                     max_speakers, threshold, median, duration_sec);
    return result;
}

std::vector<DiarSegment> decode_segments(const std::vector<float>& logits, int frames,
                                        int slots, int max_speakers, float threshold,
                                        int median, double duration_sec) {
    // Channel 0 is silence and the last channel is the non-speaker slot, so the
    // speakers are channels 1 .. slots-2.
    const int first = 1;
    const int last = std::min(slots - 2, first + std::max(max_speakers, 1) - 1);

    std::vector<DiarSegment> segments;
    for (int ch = first; ch <= last; ++ch) {
        std::vector<int> active(frames, 0);
        for (int t = 0; t < frames; ++t) {
            const float logit = logits[static_cast<size_t>(t) * slots + ch];
            const float prob = 1.0f / (1.0f + std::exp(-logit));
            active[t] = prob > threshold ? 1 : 0;
        }
        active = median_filter(active, median);

        int t = 0;
        while (t < frames) {
            if (!active[t]) { ++t; continue; }
            const int start = t;
            while (t < frames && active[t]) ++t;
            DiarSegment seg;
            seg.start_sec = start * StreamingDiarizer::kFrameSec;
            seg.end_sec = t * StreamingDiarizer::kFrameSec;
            seg.speaker_id = ch - first;
            if (duration_sec > 0.0) {
                if (seg.start_sec >= duration_sec) continue;
                seg.end_sec = std::min(seg.end_sec, duration_sec);
            }
            if (seg.end_sec > seg.start_sec) segments.push_back(seg);
        }
    }
    std::sort(segments.begin(), segments.end(), [](const DiarSegment& a, const DiarSegment& b) {
        if (a.start_sec != b.start_sec) return a.start_sec < b.start_sec;
        return a.speaker_id < b.speaker_id;
    });
    return segments;
}

void write_rttm(const std::vector<DiarSegment>& segments, const std::string& path,
                const std::string& uri) {
    std::ofstream file(path.c_str());
    if (!file) throw std::runtime_error("cannot write rttm: " + path);
    for (const DiarSegment& s : segments) {
        char line[256];
        std::snprintf(line, sizeof(line), "SPEAKER %s 1 %.3f %.3f <NA> <NA> %s_%d <NA>\n",
                      uri.c_str(), s.start_sec, s.end_sec - s.start_sec, uri.c_str(),
                      s.speaker_id);
        file << line;
    }
}
