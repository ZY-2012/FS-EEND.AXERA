// Dump the C++ frontend output so it can be diffed against the Python one.
//
// Builds without the AX Engine BSP (host or board), so the frontend can be
// verified independently of the NPU:
//
//   g++ -O2 -std=c++11 -I include -I third_party/kissfft \
//       tools/dump_features.cpp src/audio_reader.cpp src/feature_extractor.cpp \
//       third_party/kissfft/kiss_fft.c -lm -o dump_features
//   ./dump_features input.wav features.bin
//
// Output is raw float32, shape (frames, 345), row-major.
#include <cstdio>
#include <exception>
#include <fstream>
#include <vector>

#include "audio_reader.hpp"
#include "feature_extractor.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <input.wav> <features.bin>\n", argv[0]);
        return 1;
    }
    try {
        AudioData audio = read_wav_mono(argv[1]);
        if (audio.sample_rate != 8000) audio = resample_audio(audio, 8000);
        const std::vector<float> features = extract_ls_eend_features(audio);
        const size_t feat_dim = (2 * 7 + 1) * 23;

        std::ofstream out(argv[2], std::ios::binary);
        if (!out) {
            std::fprintf(stderr, "cannot write %s\n", argv[2]);
            return 1;
        }
        out.write(reinterpret_cast<const char*>(features.data()),
                  static_cast<std::streamsize>(features.size() * sizeof(float)));
        std::printf("frames=%zu dim=%zu -> %s\n", features.size() / feat_dim, feat_dim, argv[2]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
