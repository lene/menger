#include "VideoLoader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/avutil.h>
#include <libavutil/rational.h>
#include <libswscale/swscale.h>
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

double timestampSecondsForFrame(const AVStream* stream, const AVFrame* frame, int frameIndex,
    double nativeFps) {
    const int64_t frameTimestamp = frame->best_effort_timestamp != AV_NOPTS_VALUE
        ? frame->best_effort_timestamp
        : frame->pts;
    if (frameTimestamp != AV_NOPTS_VALUE && stream->time_base.den != 0) {
        const int64_t streamStart =
            stream->start_time != AV_NOPTS_VALUE ? stream->start_time : 0;
        return std::max(0.0, static_cast<double>(frameTimestamp - streamStart) *
            av_q2d(stream->time_base));
    }
    if (nativeFps > 0.0) {
        return static_cast<double>(frameIndex) / nativeFps;
    }
    return 0.0;
}

bool pixelFormatHasAlpha(AVPixelFormat pixelFormat) {
    const AVPixFmtDescriptor* descriptor = av_pix_fmt_desc_get(pixelFormat);
    return descriptor != nullptr && (descriptor->flags & AV_PIX_FMT_FLAG_ALPHA) != 0;
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

VideoFrame VideoLoader::frameAt(double timestampSeconds) {
    const int targetFrameIndex = frameIndexForTimestamp(timestampSeconds);
    if (auto frame = cachedFrame(targetFrameIndex)) {
        return *frame;
    }
    return decodeFrameAtIndex(targetFrameIndex);
}

void VideoLoader::prefetch(double timestampSeconds, int nFrames) {
    if (nFrames <= 0) {
        return;
    }

    const int startFrameIndex = frameIndexForTimestamp(timestampSeconds);
    const int requestedEndFrameIndex = startFrameIndex + std::min(nFrames, MaxCachedFrames) - 1;
    const int endFrameIndex = std::min(frameCount_ - 1, requestedEndFrameIndex);
    for (int frameIndex = startFrameIndex; frameIndex <= endFrameIndex; ++frameIndex) {
        decodeFrameAtIndex(frameIndex);
    }
}

void VideoLoader::open(const char* path) {
    if (path == nullptr || path[0] == '\0') {
        throw std::invalid_argument("Video path must not be empty");
    }
    const std::string videoPath(path);

    try {
        AVFormatContext* openedFormat = nullptr;
        int result = avformat_open_input(&openedFormat, path, nullptr, nullptr);
        if (result < 0) {
            throw std::runtime_error("Failed to open video '" + videoPath + "': " +
                avError(result));
        }
        formatContext_ = openedFormat;

        result = avformat_find_stream_info(formatContext_, nullptr);
        if (result < 0) {
            throw std::runtime_error("Failed to read stream info for video '" +
                videoPath + "': " + avError(result));
        }

        result = av_find_best_stream(formatContext_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (result < 0) {
            throw std::runtime_error("Video has no decodable video stream: " + videoPath);
        }
        videoStreamIndex_ = result;

        AVStream* stream = formatContext_->streams[videoStreamIndex_];
        const AVCodec* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
        if (decoder == nullptr) {
            throw std::runtime_error("No decoder found for video stream: " + videoPath);
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

        packet_ = av_packet_alloc();
        decodedFrame_ = av_frame_alloc();
        if (packet_ == nullptr || decodedFrame_ == nullptr) {
            throw std::runtime_error("Failed to allocate video decode buffers");
        }

        width_ = codecContext_->width;
        height_ = codecContext_->height;
        durationSeconds_ = secondsFromStreamDuration(formatContext_, stream);
        nativeFps_ = fpsForStream(formatContext_, stream);
        frameCount_ = frameCountForStream(stream, durationSeconds_, nativeFps_);

        if (width_ <= 0 || height_ <= 0) {
            throw std::runtime_error("Video has invalid dimensions: " + videoPath);
        }
        if (durationSeconds_ <= 0.0) {
            throw std::runtime_error("Video has invalid duration: " + videoPath);
        }
        if (frameCount_ <= 0) {
            throw std::runtime_error("Video has no frames: " + videoPath);
        }
    } catch (...) {
        close();
        throw;
    }
}

void VideoLoader::close() {
    frameCache_.clear();
    cacheOrder_.clear();
    lastDecodedFrameIndex_.reset();
    if (swsContext_ != nullptr) {
        sws_freeContext(swsContext_);
        swsContext_ = nullptr;
    }
    if (decodedFrame_ != nullptr) {
        av_frame_free(&decodedFrame_);
    }
    if (packet_ != nullptr) {
        av_packet_free(&packet_);
    }
    if (codecContext_ != nullptr) {
        avcodec_free_context(&codecContext_);
    }
    if (formatContext_ != nullptr) {
        avformat_close_input(&formatContext_);
    }
    videoStreamIndex_ = -1;
    width_ = 0;
    height_ = 0;
    durationSeconds_ = 0.0;
    frameCount_ = 0;
    nativeFps_ = 0.0;
    nextFrameIndex_ = 0;
    inputEnded_ = false;
    decoderEnded_ = false;
}

AVStream* VideoLoader::videoStream() const {
    if (formatContext_ == nullptr || videoStreamIndex_ < 0 ||
        videoStreamIndex_ >= static_cast<int>(formatContext_->nb_streams)) {
        throw std::logic_error("Video stream is not open");
    }
    return formatContext_->streams[videoStreamIndex_];
}

int VideoLoader::frameIndexForTimestamp(double timestampSeconds) const {
    if (!std::isfinite(timestampSeconds)) {
        throw std::invalid_argument("Video timestamp must be finite");
    }
    if (frameCount_ <= 0 || durationSeconds_ <= 0.0) {
        throw std::logic_error("Video metadata is invalid");
    }

    const double fps = nativeFps_ > 0.0
        ? nativeFps_
        : static_cast<double>(frameCount_) / durationSeconds_;
    if (fps <= 0.0) {
        throw std::logic_error("Video fps is invalid");
    }

    const double boundedTimestamp = std::clamp(timestampSeconds, 0.0, durationSeconds_);
    const double rawFrameIndex = std::floor(boundedTimestamp * fps + 1.0e-9);
    const int frameIndex = static_cast<int>(
        std::min(rawFrameIndex, static_cast<double>(frameCount_ - 1))
    );
    return std::max(0, frameIndex);
}

void VideoLoader::resetDecoderToStart() {
    AVStream* stream = videoStream();
    const int64_t streamStart =
        stream->start_time != AV_NOPTS_VALUE ? stream->start_time : 0;
    const int result = av_seek_frame(
        formatContext_,
        videoStreamIndex_,
        streamStart,
        AVSEEK_FLAG_BACKWARD
    );
    if (result < 0) {
        throw std::runtime_error("Failed to seek video to start: " + avError(result));
    }

    avcodec_flush_buffers(codecContext_);
    av_packet_unref(packet_);
    av_frame_unref(decodedFrame_);
    nextFrameIndex_ = 0;
    inputEnded_ = false;
    decoderEnded_ = false;
    lastDecodedFrameIndex_.reset();
}

void VideoLoader::sendNextPacket() {
    if (inputEnded_) {
        return;
    }

    while (true) {
        const int readResult = av_read_frame(formatContext_, packet_);
        if (readResult == AVERROR_EOF) {
            const int sendResult = avcodec_send_packet(codecContext_, nullptr);
            if (sendResult < 0 && sendResult != AVERROR_EOF) {
                throw std::runtime_error("Failed to flush video decoder: " +
                    avError(sendResult));
            }
            inputEnded_ = true;
            return;
        }
        if (readResult < 0) {
            throw std::runtime_error("Failed to read video packet: " + avError(readResult));
        }

        if (packet_->stream_index != videoStreamIndex_) {
            av_packet_unref(packet_);
            continue;
        }

        const int sendResult = avcodec_send_packet(codecContext_, packet_);
        av_packet_unref(packet_);
        if (sendResult < 0) {
            throw std::runtime_error("Failed to send video packet to decoder: " +
                avError(sendResult));
        }
        return;
    }
}

std::optional<VideoFrame> VideoLoader::decodeNextFrame() {
    if (decoderEnded_) {
        return std::nullopt;
    }

    while (true) {
        const int receiveResult = avcodec_receive_frame(codecContext_, decodedFrame_);
        if (receiveResult == 0) {
            const int frameIndex = nextFrameIndex_++;
            VideoFrame frame = convertFrame(decodedFrame_, frameIndex);
            av_frame_unref(decodedFrame_);
            cacheFrame(frameIndex, frame);
            lastDecodedFrameIndex_ = frameIndex;
            return frame;
        }
        if (receiveResult == AVERROR_EOF) {
            decoderEnded_ = true;
            return std::nullopt;
        }
        if (receiveResult != AVERROR(EAGAIN)) {
            throw std::runtime_error("Failed to decode video frame: " +
                avError(receiveResult));
        }
        if (inputEnded_) {
            decoderEnded_ = true;
            return std::nullopt;
        }
        sendNextPacket();
    }
}

VideoFrame VideoLoader::decodeFrameAtIndex(int targetFrameIndex) {
    if (targetFrameIndex < 0) {
        throw std::invalid_argument("Video frame index must not be negative");
    }
    if (auto frame = cachedFrame(targetFrameIndex)) {
        return *frame;
    }
    if (targetFrameIndex < nextFrameIndex_) {
        resetDecoderToStart();
    }

    while (nextFrameIndex_ <= targetFrameIndex) {
        if (!decodeNextFrame()) {
            break;
        }
    }

    if (auto frame = cachedFrame(targetFrameIndex)) {
        return *frame;
    }
    if (lastDecodedFrameIndex_) {
        if (auto frame = cachedFrame(*lastDecodedFrameIndex_)) {
            return *frame;
        }
    }
    throw std::runtime_error("Failed to decode requested video frame");
}

VideoFrame VideoLoader::convertFrame(const AVFrame* frame, int frameIndex) {
    if (frame->width <= 0 || frame->height <= 0) {
        throw std::runtime_error("Decoded video frame has invalid dimensions");
    }

    const auto sourcePixelFormat = static_cast<AVPixelFormat>(frame->format);
    swsContext_ = sws_getCachedContext(
        swsContext_,
        frame->width,
        frame->height,
        sourcePixelFormat,
        width_,
        height_,
        AV_PIX_FMT_RGBA,
        SWS_BILINEAR,
        nullptr,
        nullptr,
        nullptr
    );
    if (swsContext_ == nullptr) {
        throw std::runtime_error("Failed to create video RGBA conversion context");
    }

    const size_t pixelCount = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    if (pixelCount > std::numeric_limits<size_t>::max() / 4) {
        throw std::runtime_error("Decoded video frame is too large");
    }
    std::vector<std::uint8_t> rgba(pixelCount * 4);
    uint8_t* destinationData[4] = { rgba.data(), nullptr, nullptr, nullptr };
    int destinationLineSize[4] = { width_ * 4, 0, 0, 0 };

    const int convertedHeight = sws_scale(
        swsContext_,
        frame->data,
        frame->linesize,
        0,
        frame->height,
        destinationData,
        destinationLineSize
    );
    if (convertedHeight != height_) {
        throw std::runtime_error("Failed to convert complete video frame to RGBA");
    }

    if (!pixelFormatHasAlpha(sourcePixelFormat)) {
        for (size_t alphaIndex = 3; alphaIndex < rgba.size(); alphaIndex += 4) {
            rgba[alphaIndex] = 255;
        }
    }

    return VideoFrame{
        width_,
        height_,
        timestampSecondsForFrame(videoStream(), frame, frameIndex, nativeFps_),
        std::move(rgba)
    };
}

void VideoLoader::cacheFrame(int frameIndex, const VideoFrame& frame) {
    const bool isNewFrame = frameCache_.find(frameIndex) == frameCache_.end();
    frameCache_[frameIndex] = frame;
    if (isNewFrame) {
        cacheOrder_.push_back(frameIndex);
    }

    while (cacheOrder_.size() > MaxCachedFrames) {
        frameCache_.erase(cacheOrder_.front());
        cacheOrder_.pop_front();
    }
}

std::optional<VideoFrame> VideoLoader::cachedFrame(int frameIndex) const {
    const auto found = frameCache_.find(frameIndex);
    if (found == frameCache_.end()) {
        return std::nullopt;
    }
    return found->second;
}

} // namespace menger::geometry
