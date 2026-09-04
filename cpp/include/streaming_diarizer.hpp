#pragma once
#include <string>
#include <vector>

#include "audio_reader.hpp"
#include "ax_engine_runner.hpp"

struct DiarSegment {
    double start_sec = 0.0;
    double end_sec = 0.0;
    int speaker_id = 0;
};

struct DiarResult {
    std::vector<DiarSegment> segments;
    std::vector<float> logits;   // (emitted_frames, slots) row-major
    int slots = 0;               // model output channels: ch0 silence, ch1..slots-2 speakers
    int frames = 0;              // feature frames fed to the encoder
    int emitted_frames = 0;      // frames that produced a prediction
    double duration_sec = 0.0;
    double latency_ms_total = 0.0;
    double latency_ms_per_frame = 0.0;
    double rtf = 0.0;
};

// Frame-synchronous LS-EEND runner, line-for-line equivalent to
// python/ls_eend_sdk/session.py.
class StreamingDiarizer {
public:
    static constexpr int kConvDelay = 9;  // StreamingConv1d emits from its 10th call
    static constexpr int kEncLayers = 4;
    static constexpr int kDecLayers = 2;
    static constexpr double kFrameSec = 0.1;

    explicit StreamingDiarizer(const std::string& model_path);

    // Output channel count, read from the model: each LS-EEND release is trained
    // with a different max_speakers (simu 8 -> 10 channels, AMI 4 -> 6,
    // CALLHOME 7 -> 9, DIHARD 10 -> 12).
    int slots() const { return slots_; }

    // features: (frames, 345) row-major log-mel, as produced by
    // extract_ls_eend_features().
    DiarResult run(const std::vector<float>& features, int frames, double duration_sec,
                   int max_speakers = 8, float threshold = 0.5f, int median = 11);

private:
    void reset();

    AxEngineRunner runner_;
    // Retention running means are kept here in FP32. The graph returns only the
    // bounded per-frame increment, so accumulating on the host keeps quantization
    // error per-frame instead of letting it compound through the recurrence.
    std::vector<std::vector<float>> enc_mean_, dec_mean_;
    std::vector<std::vector<float>> enc_conv_;
    std::vector<float> conv_cache_;
    int enc_t_ = 0;
    int dec_t_ = 0;
    int slots_ = 0;
};

std::vector<DiarSegment> decode_segments(const std::vector<float>& logits, int frames,
                                        int slots, int max_speakers, float threshold,
                                        int median, double duration_sec);
void write_rttm(const std::vector<DiarSegment>& segments, const std::string& path,
                const std::string& uri);
