#pragma once
#include <vector>
#include "audio_reader.hpp"

// LS-EEND frontend: 8 kHz, 25 ms window (200), 10 ms hop (80),
// 1024-point STFT, 23 log-mel bins, +/-7 frame splice, subsample by 10.
std::vector<float> extract_ls_eend_features(const AudioData& audio);
