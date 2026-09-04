#include "audio_reader.hpp"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
uint16_t read_u16(std::ifstream& f) { uint8_t b[2]; f.read(reinterpret_cast<char*>(b),2); return b[0] | (b[1]<<8); }
uint32_t read_u32(std::ifstream& f) { uint8_t b[4]; f.read(reinterpret_cast<char*>(b),4); return b[0] | (b[1]<<8) | (b[2]<<16) | (b[3]<<24); }
}

AudioData read_wav_mono(const std::string& path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    if (!f) throw std::runtime_error("cannot open wav: " + path);
    char riff[4], wave[4]; f.read(riff,4); (void)read_u32(f); f.read(wave,4);
    if (std::string(riff,4)!="RIFF" || std::string(wave,4)!="WAVE") throw std::runtime_error("not RIFF/WAVE");
    uint16_t format=0, channels=0, bits=0; uint32_t rate=0; std::vector<uint8_t> pcm;
    while (f && !f.eof()) {
        char id[4]; f.read(id,4); if (!f) break; uint32_t n=read_u32(f); std::string tag(id,4);
        if (tag=="fmt ") { format=read_u16(f); channels=read_u16(f); rate=read_u32(f); (void)read_u32(f); (void)read_u16(f); bits=read_u16(f); if(n>16)f.seekg(n-16,std::ios::cur); }
        else if(tag=="data") { pcm.resize(n); f.read(reinterpret_cast<char*>(pcm.data()),n); }
        else f.seekg(n,std::ios::cur);
        if(n&1)f.seekg(1,std::ios::cur);
    }
    if(format!=1 || bits!=16 || channels<1 || rate==0 || pcm.empty()) throw std::runtime_error("only PCM16 WAV supported");
    size_t frames=pcm.size()/(channels*2); AudioData out; out.sample_rate=(int)rate; out.samples.resize(frames);
    for(size_t i=0;i<frames;++i) { int sum=0; for(int c=0;c<channels;++c){size_t p=(i*channels+c)*2; int16_t v=(int16_t)(pcm[p]|(pcm[p+1]<<8)); sum+=v;} out.samples[i]=(float)sum/(channels*32768.0f); }
    return out;
}

AudioData resample_linear(const AudioData& input, int target_rate) {
    if(input.sample_rate==target_rate)return input;
    AudioData out; out.sample_rate=target_rate; size_t n=(size_t)std::llround(input.samples.size()*(double)target_rate/input.sample_rate); out.samples.resize(n); double scale=(double)input.sample_rate/target_rate;
    for(size_t i=0;i<n;++i){double x=i*scale;size_t j=(size_t)x;double a=x-j;out.samples[i]=(j+1<input.samples.size())?(float)((1-a)*input.samples[j]+a*input.samples[j+1]):input.samples.back();}
    return out;
}
