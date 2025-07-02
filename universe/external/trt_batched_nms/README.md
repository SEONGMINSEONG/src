# trt_batched_nms

ROS 2 package for TensorRT batched NMS plugin developed by NVIDIA and modified
by mmdeploy.

**Original source code:**

- https://github.com/NVIDIA/TensorRT/tree/release/10.3/plugin/batchedNMSPlugin
- https://github.com/open-mmlab/mmdeploy/tree/main/csrc/mmdeploy/backend_ops/tensorrt/batched_nms

## Compile the package:

```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## Usage

After compiling the package, shared library `libtrt_batched_nms_plugin.so` will be
generated in the `install/trt_batched_nms/lib` directory.

You can use following logic to use generated shared library in cpp code:

```cpp
const std::string libPath = "/path/to/libtrt_batched_nms_plugin.so";
void *handle = dlopen(libPath, RTLD_LAZY);
```

Also, you can use following example in your `launch` or `config` files to use
generated shared library as parameter in your ROS 2 packages.

- `"$(find-pkg-prefix trt_batched_nms)/lib/libtrt_batched_nms_plugin.so"`