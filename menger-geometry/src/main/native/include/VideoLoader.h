#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct AVCodecContext;
struct AVFormatContext;

namespace menger::geometry {

struct VideoFrame {
    int width;
    int height;
    double timestampSeconds;
    std::vector<std::uint8_t> rgba;
};

class VideoLoader {
public:
    explicit VideoLoader(const char* path);
    ~VideoLoader();

    VideoLoader(const VideoLoader&) = delete;
    VideoLoader& operator=(const VideoLoader&) = delete;

    VideoLoader(VideoLoader&&) = delete;
    VideoLoader& operator=(VideoLoader&&) = delete;

    int width() const;
    int height() const;
    double durationSeconds() const;
    int frameCount() const;
    double nativeFps() const;

    VideoFrame frameAt(double timestampSeconds);
    void prefetch(double timestampSeconds, int nFrames);

private:
    AVFormatContext* formatContext_ = nullptr;
    AVCodecContext* codecContext_ = nullptr;
    int videoStreamIndex_ = -1;
    int width_ = 0;
    int height_ = 0;
    double durationSeconds_ = 0.0;
    int frameCount_ = 0;
    double nativeFps_ = 0.0;

    void open(const char* path);
    void close();
};

} // namespace menger::geometry
