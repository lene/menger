#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct AVCodecContext;
struct AVFrame;
struct AVFormatContext;
struct AVPacket;
struct AVStream;
struct SwsContext;

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
    static constexpr int MaxCachedFrames = 8;

    AVFormatContext* formatContext_ = nullptr;
    AVCodecContext* codecContext_ = nullptr;
    AVPacket* packet_ = nullptr;
    AVFrame* decodedFrame_ = nullptr;
    SwsContext* swsContext_ = nullptr;
    int videoStreamIndex_ = -1;
    int width_ = 0;
    int height_ = 0;
    double durationSeconds_ = 0.0;
    int frameCount_ = 0;
    double nativeFps_ = 0.0;
    int nextFrameIndex_ = 0;
    bool inputEnded_ = false;
    bool decoderEnded_ = false;
    std::optional<int> lastDecodedFrameIndex_;
    std::unordered_map<int, VideoFrame> frameCache_;
    std::deque<int> cacheOrder_;

    void open(const char* path);
    void close();
    AVStream* videoStream() const;
    int frameIndexForTimestamp(double timestampSeconds) const;
    void resetDecoderToStart();
    void sendNextPacket();
    std::optional<VideoFrame> decodeNextFrame();
    VideoFrame decodeFrameAtIndex(int targetFrameIndex);
    VideoFrame convertFrame(const AVFrame* frame, int frameIndex);
    void cacheFrame(int frameIndex, const VideoFrame& frame);
    std::optional<VideoFrame> cachedFrame(int frameIndex) const;
};

} // namespace menger::geometry
