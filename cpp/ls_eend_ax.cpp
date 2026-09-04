// LS-EEND streaming speaker diarization on AX650N: wav -> RTTM.
//
// Inference logic mirrors python/ls_eend_sdk (same frontend, same warmup gating,
// same FP32 host accumulation, same post-processing), so the two produce the
// same RTTM for the same input.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "audio_reader.hpp"
#include "feature_extractor.hpp"
#include "streaming_diarizer.hpp"

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s --model <streaming_step.axmodel> --wav <input.wav>\n"
                 "          [--rttm <out.rttm>] [--max-speakers N] [--threshold F]\n"
                 "          [--median N]\n\n"
                 "  --model         quantized axmodel (AX650/NPU3)\n"
                 "  --wav           input audio, any sample rate (resampled to 8 kHz)\n"
                 "  --rttm          output RTTM path (default: <wav>.rttm)\n"
                 "  --max-speakers  keep the first N of 8 speaker channels (default 8)\n"
                 "  --threshold     activity threshold on the sigmoid (default 0.5)\n"
                 "  --median        median filter length in frames (default 11)\n",
                 argv0);
}

std::string stem_of(const std::string& path) {
    size_t slash = path.find_last_of('/');
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    return (dot == std::string::npos) ? base : base.substr(0, dot);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, wav_path, rttm_path;
    int max_speakers = 8;
    int median = 11;
    float threshold = 0.5f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_next = (i + 1 < argc);
        if (arg == "--model" && has_next) model_path = argv[++i];
        else if (arg == "--wav" && has_next) wav_path = argv[++i];
        else if (arg == "--rttm" && has_next) rttm_path = argv[++i];
        else if (arg == "--max-speakers" && has_next) max_speakers = std::atoi(argv[++i]);
        else if (arg == "--threshold" && has_next) threshold = std::atof(argv[++i]);
        else if (arg == "--median" && has_next) median = std::atoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", arg.c_str()); usage(argv[0]); return 1; }
    }
    if (model_path.empty() || wav_path.empty()) { usage(argv[0]); return 1; }
    if (rttm_path.empty()) rttm_path = stem_of(wav_path) + ".rttm";

    try {
        AudioData audio = read_wav_mono(wav_path);
        const double duration = audio.samples.size() / static_cast<double>(audio.sample_rate);
        if (audio.sample_rate != 8000) audio = resample_audio(audio, 8000);

        std::vector<float> features = extract_ls_eend_features(audio);
        const int feat_dim = (2 * 7 + 1) * 23;  // 345
        const int frames = static_cast<int>(features.size() / feat_dim);
        std::printf("audio: %.2f s -> %d feature frames\n", duration, frames);

        StreamingDiarizer diarizer(model_path);
        DiarResult result =
            diarizer.run(features, frames, duration, max_speakers, threshold, median);

        write_rttm(result.segments, rttm_path, stem_of(wav_path));

        std::printf("frames          : %d (emitted %d)\n", result.frames, result.emitted_frames);
        std::printf("segments        : %zu\n", result.segments.size());
        std::printf("latency         : %.1f ms total, %.3f ms/frame\n", result.latency_ms_total,
                    result.latency_ms_per_frame);
        std::printf("RTF             : %.4f\n", result.rtf);
        std::printf("rttm            : %s\n", rttm_path.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
