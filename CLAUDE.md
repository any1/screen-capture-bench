# Screen capturing benchmark

This is a benchmark for screen capturing on Linux.

## Architecture

 * Language: C
 * Style: Linux Kernel Style, with exceptions for user space
 * Build: meson
 * Dependencies:
   - libgbm
   - libdrm
   - vulkan
   - json-c

### Buffer allocation

The single render buffer is allocated using Vulkan. The VkImage render target
is set up using a specific modifier as defined in drm_fourcc.h, including
linear.

The shadow buffer, if used, shall be host cached if possible.

Both buffers should be pre-allocated and re-used.

## Method

Count how many frames can be rendered and captured within the test duration that
shall be 1 second by default. The frames shall be rendered to be played back at
60 frames per second.

 1. Render a rotating triangle using vulkan graphics
 2. Copy the rendered frame to a buffer allocated using malloc.
 3. Encode the frame to mjpeg via turbo jpeg (appended jpeg images).

The third step should be turned on using an option. It is only for verification.
The memcpy to the CPU allocated buffer must not be optimised out. The output
file shall be named according the the test combination.

Copying the rendered frame to the host (step 2) may require the use of a shadow
buffer, but only when using Vulkan mapping.

When mapping using gbm_bo_map, a shadow buffer shall never be used as this is
handled internally by gbm.

## Tests

We want to test all combination of...

 * Each available GPU
 * Each available buffer modifier
 * Use of shadow buffer vs. no shadow buffer.
 * Mapping using GBM vs. mapping via Vulkan.

Impossible combinations shall be skipped.

Each test shall be shown on stderr as it happens, and all the result should be
printed as json to stdout in the end.
