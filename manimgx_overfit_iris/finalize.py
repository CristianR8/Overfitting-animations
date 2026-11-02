"""Preserve ManimGX's static ending in a constant-frame-rate H.264 MP4."""
import argparse
from fractions import Fraction
import av


def finalize(source, destination, fps, duration):
    frames_needed = round(fps*duration)
    written = 0
    with av.open(source) as source_video:
        video = source_video.streams.video[0]
        with av.open(destination, 'w', options={'movflags': '+faststart'}) as output:
            stream = output.add_stream('libx264', rate=fps)
            stream.width, stream.height = video.width, video.height
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '16', 'preset': 'medium', 'tune': 'animation'}
            output.metadata['title'] = 'CVAIL Talk Series · Overfitting'

            def emit(frame, index):
                frame.pts = index
                frame.time_base = Fraction(1, fps)
                frame.pict_type = av.video.frame.PictureType.NONE
                for packet in stream.encode(frame):
                    output.mux(packet)

            previous = None
            for frame in source_video.decode(video=0):
                timestamp = float(frame.time)
                if previous is not None:
                    while written < frames_needed and written/fps < timestamp-1e-8:
                        emit(previous, written)
                        written += 1
                previous = frame
            if previous is None:
                raise ValueError('The ManimGX render contains no video frames')
            while written < frames_needed:
                emit(previous, written)
                written += 1
            for packet in stream.encode():
                output.mux(packet)
    print(f'{destination}: {written} frames, {fps} fps, {duration:g} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--duration', type=float, required=True)
    args = parser.parse_args()
    finalize(args.source, args.destination, args.fps, args.duration)
