#include "VideoLoader.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/rational.h>
}

namespace menger::geometry {
namespace {

std::string avError(int code) {
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(code, buffer, sizeof(buffer));
    return std::string(buffer);
}

double secondsFromStreamDuration(const AVFormatContext* formatContext, const AVStream* stream) {
    if (stream->duration != AV_NOPTS_VALUE && stream->time_base.den != 0) {
        return static_cast<double>(stream->duration) * av_q2d(stream->time_base);
    }
    if (formatContext->duration != AV_NOPTS_VALUE) {
        return static_cast<double>(formatContext->duration) / AV_TIME_BASE;
    }
    return 0.0;
}

double fpsForStream(AVFormatContext* formatContext, AVStream* stream) {
    AVRational guessed = av_guess_frame_rate(formatContext, stream, nullptr);
    if (guessed.num > 0 && guessed.den > 0) {
        return av_q2d(guessed);
    }
    if (stream->avg_frame_rate.num > 0 && stream->avg_frame_rate.den > 0) {
        return av_q2d(stream->avg_frame_rate);
    }
    return 0.0;
}

int frameCountForStream(const AVStream* stream, double durationSeconds, double fps) {
    if (stream->nb_frames > 0) {
        const int64_t bounded =
            std::min<int64_t>(stream->nb_frames, std::numeric_limits<int>::max());
        return static_cast<int>(bounded);
    }
    if (durationSeconds > 0.0 && fps > 0.0) {
        const double estimated = std::round(durationSeconds * fps);
        if (estimated > 0.0) {
            const double bounded =
                std::min(estimated, static_cast<double>(std::numeric_limits<int>::max()));
            return static_cast<int>(bounded);
        }
    }
    return 0;
}

} // namespace

VideoLoader::VideoLoader(const char* path) {
    open(path);
}

VideoLoader::~VideoLoader() {
    close();
}

int VideoLoader::width() const {
    return width_;
}

int VideoLoader::height() const {
    return height_;
}

double VideoLoader::durationSeconds() const {
    return durationSeconds_;
}

int VideoLoader::frameCount() const {
    return frameCount_;
}

double VideoLoader::nativeFps() const {
    return nativeFps_;
}

VideoFrame VideoLoader::frameAt(double /*timestampSeconds*/) {
    throw std::logic_error("Video frame decode is implemented in Sprint 27.2");
}

void VideoLoader::prefetch(double /*timestampSeconds*/, int /*nFrames*/) {
}

void VideoLoader::open(const char* path) {
    if (path == nullptr || path[0] == '\0') {
        throw std::invalid_argument("Video path must not be empty");
    }

    AVFormatContext* openedFormat = nullptr;
    int result = avformat_open_input(&openedFormat, path, nullptr, nullptr);
    if (result < 0) {
        throw std::runtime_error("Failed to open video '" + std::string(path) + "': " +
            avError(result));
    }
    formatContext_ = openedFormat;

    result = avformat_find_stream_info(formatContext_, nullptr);
    if (result < 0) {
        throw std::runtime_error("Failed to read stream info for video '" +
            std::string(path) + "': " + avError(result));
    }

    result = av_find_best_stream(formatContext_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (result < 0) {
        throw std::runtime_error("Video has no decodable video stream: " + std::string(path));
    }
    videoStreamIndex_ = result;

    AVStream* stream = formatContext_->streams[videoStreamIndex_];
    const AVCodec* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
    if (decoder == nullptr) {
        throw std::runtime_error("No decoder found for video stream: " + std::string(path));
    }

    codecContext_ = avcodec_alloc_context3(decoder);
    if (codecContext_ == nullptr) {
        throw std::runtime_error("Failed to allocate video decoder context");
    }

    result = avcodec_parameters_to_context(codecContext_, stream->codecpar);
    if (result < 0) {
        throw std::runtime_error("Failed to copy video codec parameters: " + avError(result));
    }

    result = avcodec_open2(codecContext_, decoder, nullptr);
    if (result < 0) {
        throw std::runtime_error("Failed to open video decoder: " + avError(result));
    }

    width_ = codecContext_->width;
    height_ = codecContext_->height;
    durationSeconds_ = secondsFromStreamDuration(formatContext_, stream);
    nativeFps_ = fpsForStream(formatContext_, stream);
    frameCount_ = frameCountForStream(stream, durationSeconds_, nativeFps_);

    if (width_ <= 0 || height_ <= 0) {
        throw std::runtime_error("Video has invalid dimensions: " + std::string(path));
    }
    if (durationSeconds_ <= 0.0) {
        throw std::runtime_error("Video has invalid duration: " + std::string(path));
    }
    if (frameCount_ <= 0) {
        throw std::runtime_error("Video has no frames: " + std::string(path));
    }
}

void VideoLoader::close() {
    if (codecContext_ != nullptr) {
        avcodec_free_context(&codecContext_);
    }
    if (formatContext_ != nullptr) {
        avformat_close_input(&formatContext_);
    }
    videoStreamIndex_ = -1;
}

} // namespace menger::geometry
