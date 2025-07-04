// Copyright 2021 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AUTOWARE__LIDAR_CENTERPOINT__PREPROCESS__VOXEL_GENERATOR_HPP_
#define AUTOWARE__LIDAR_CENTERPOINT__PREPROCESS__VOXEL_GENERATOR_HPP_

#include "autoware/lidar_centerpoint/centerpoint_config.hpp"
#include "autoware/lidar_centerpoint/cuda_utils.hpp"
#include "autoware/lidar_centerpoint/preprocess/pointcloud_densification.hpp"
#include "autoware/lidar_centerpoint/preprocess/preprocess_kernel.hpp"

#include <cuda_blackboard/cuda_pointcloud2.hpp>

#include <memory>
#include <vector>

namespace autoware::lidar_centerpoint
{
constexpr std::size_t AFF_MAT_SIZE = 16;  // 4x4 matrix
class VoxelGeneratorTemplate
{
public:
  virtual ~VoxelGeneratorTemplate() = default;
  explicit VoxelGeneratorTemplate(
    const DensificationParam & param, const CenterPointConfig & config, cudaStream_t & stream);

  virtual std::size_t generateSweepPoints(float * points_d) = 0;

  bool enqueuePointCloud(
    const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & input_pointcloud_msg_ptr,
    const tf2_ros::Buffer & tf_buffer);

protected:
  std::unique_ptr<PointCloudDensification> pd_ptr_{nullptr};
  std::unique_ptr<PreprocessCuda> pre_ptr_{nullptr};
  cuda::unique_ptr<float[]> affine_past2current_d_{nullptr};

  CenterPointConfig config_;
  std::array<float, 6> range_;
  std::array<int, 3> grid_size_;
  std::array<float, 3> recip_voxel_size_;
  cudaStream_t stream_;
};

class VoxelGenerator : public VoxelGeneratorTemplate
{
public:
  using VoxelGeneratorTemplate::VoxelGeneratorTemplate;

  std::size_t generateSweepPoints(float * points_d) override;
};

}  // namespace autoware::lidar_centerpoint

#endif  // AUTOWARE__LIDAR_CENTERPOINT__PREPROCESS__VOXEL_GENERATOR_HPP_
