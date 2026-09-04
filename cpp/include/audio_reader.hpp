#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct AudioData {
    int sample_rate = 0;
    std::vector<float> samples;
};

AudioData read_wav_mono(const std::string& path);
AudioData resample_linear(const AudioData& input, int target_rate);
