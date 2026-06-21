# Screen Capturing Benchmark

This is a simple benchmark that renders and copies as many frames as possible
within a given duration (1 second by default), with the goal of finding out what
is the most efficient and/or performant way to capture video frames on different
devices.

The way that memory is shared between the GPU and the CPU differs between
architectures, so I'd like to get some data to figure out how to decide on a
method when running on different devices.

By measuring the total throughput over a given time frame, we should be able to
find the capturing methods that take the least time in total (CPU and GPU
combined). It might be possible to increase the total throughput by applying
parallelism on whichever device is less busy, but that's not the goal of the
experiment.

## Building

```
meson setup build
ninja -C build
```

## Running

```
# Generate report
./build/screen-capture-bench | ./scripts/html_report.py >report.html

# Open in browser
xdg-open report.html
```
