// Copyright 2023 TIER IV, Inc.
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

#include "yabloc_image_processing/segment_filter/segment_filter.hpp"

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <yabloc_common/cv_decompress.hpp>
#include <yabloc_common/pub_sub.hpp>

#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <limits>
#include <set>
#include <utility>

namespace yabloc::segment_filter
{
SegmentFilter::SegmentFilter(const rclcpp::NodeOptions & options)
: Node("segment_filter", options),
  image_size_(static_cast<int>(declare_parameter<int>("image_size"))),
  max_range_(static_cast<float>(declare_parameter<float>("max_range"))),
  min_segment_length_(static_cast<float>(declare_parameter<float>("min_segment_length"))),
  max_segment_distance_(static_cast<float>(declare_parameter<float>("max_segment_distance"))),
  max_lateral_distance_(static_cast<float>(declare_parameter<float>("max_lateral_distance"))),
  info_(this),
  synchro_subscriber_(this, "~/input/line_segments_cloud", "~/input/graph_segmented"),
  tf_subscriber_(this->get_clock())
{
  using std::placeholders::_1;
  using std::placeholders::_2;
  auto cb = std::bind(&SegmentFilter::execute, this, _1, _2);
  synchro_subscriber_.set_callback(std::move(cb));

  pub_projected_cloud_ =
    create_publisher<PointCloud2>("~/output/projected_line_segments_cloud", 10);
  pub_debug_cloud_ = create_publisher<PointCloud2>("~/debug/debug_line_segments", 10);
  pub_image_ = create_publisher<Image>("~/debug/projected_image", 10);
}

cv::Point2i SegmentFilter::to_cv_point(const Eigen::Vector3f & v) const
{
  cv::Point2i pt;
  pt.x = static_cast<int>(
    -v.y() / max_range_ * static_cast<float>(image_size_) * 0.5f +
    static_cast<float>(image_size_) / 2.0);
  pt.y = static_cast<int>(
    -v.x() / max_range_ * static_cast<float>(image_size_) * 0.5f + static_cast<float>(image_size_));
  return pt;
}

bool SegmentFilter::define_project_func()
{
  if (project_func_) return true;
  
  if (info_.is_camera_info_nullopt()) {
    RCLCPP_WARN(get_logger(), "Camera info is null!");
    return false;
  }
  Eigen::Matrix3f intrinsic_inv = info_.intrinsic().inverse();
  
  std::optional<Eigen::Affine3f> camera_extrinsic =
    tf_subscriber_(info_.get_frame_id(), "base_link");
  if (!camera_extrinsic.has_value()) {
    RCLCPP_WARN(get_logger(), "Camera extrinsic not available!");
    return false;
  }
  
  const Eigen::Vector3f t = camera_extrinsic->translation();
  const Eigen::Quaternionf q(camera_extrinsic->rotation());
  
  // HJK: 카메라 파라미터 출력
  // RCLCPP_INFO(get_logger(), "Camera setup - translation: (%.3f, %.3f, %.3f)", t.x(), t.y(), t.z());
  // RCLCPP_INFO(get_logger(), "Camera intrinsic:\n[%.1f, %.1f, %.1f]\n[%.1f, %.1f, %.1f]\n[%.1f, %.1f, %.1f]",
  //             info_.intrinsic()(0,0), info_.intrinsic()(0,1), info_.intrinsic()(0,2),
  //             info_.intrinsic()(1,0), info_.intrinsic()(1,1), info_.intrinsic()(1,2),
  //             info_.intrinsic()(2,0), info_.intrinsic()(2,1), info_.intrinsic()(2,2));
  
  // HJK: 디버그 카운터 추가
  std::shared_ptr<int> debug_count = std::make_shared<int>(0);
  
  project_func_ = [intrinsic_inv, q, t, this, debug_count](const Eigen::Vector3f & u) -> std::optional<Eigen::Vector3f> {
    Eigen::Vector3f u3(u.x(), u.y(), 1);
    Eigen::Vector3f u_bearing = (q * intrinsic_inv * u3).normalized();
    
    // HJK: 처음 3개만 상세 로그 출력
    if (*debug_count < 3) {
      // RCLCPP_INFO(get_logger(), "=== Projection attempt %d ===", *debug_count);
      // RCLCPP_INFO(get_logger(), "  pixel: (%.1f, %.1f)", u.x(), u.y());
      // RCLCPP_INFO(get_logger(), "  u_bearing: (%.3f, %.3f, %.3f)", u_bearing.x(), u_bearing.y(), u_bearing.z());
      // RCLCPP_INFO(get_logger(), "  camera t: (%.3f, %.3f, %.3f)", t.x(), t.y(), t.z());
    }
    
    // HJK: 조건 완화
    if (u_bearing.z() > 0.99) {
      if (*debug_count < 3) {
        // RCLCPP_WARN(get_logger(), "  -> Rejected: u_bearing.z() = %.3f > 0.99", u_bearing.z());
      }
      return std::nullopt;
    }
    
    // HJK: 0으로 나누기 방지
    if (std::abs(u_bearing.z()) < 0.001f) {
      if (*debug_count < 3) {
        // RCLCPP_WARN(get_logger(), "  -> Rejected: u_bearing.z() too small = %.6f", u_bearing.z());
      }
      return std::nullopt;
    }
    
    // HJK: 부호 수정 - 수평 카메라는 양수 u_bearing.z()를 가지므로
    float u_distance = t.z() / u_bearing.z();  // 원본: -t.z() / u_bearing.z()
    
    if (*debug_count < 3) {
      // RCLCPP_INFO(get_logger(), "  u_distance: %.3f", u_distance);
    }
    
    // HJK: 음수 조건 제거, 너무 먼 거리만 체크
    if (u_distance > 100.0f) {  // 원본: u_distance < 0 || u_distance > 100.0f
      if (*debug_count < 3) {
        // RCLCPP_WARN(get_logger(), "  -> Rejected: u_distance too far: %.3f", u_distance);
      }
      return std::nullopt;
    }
    
    Eigen::Vector3f v;
    v.x() = t.x() + u_bearing.x() * u_distance;
    v.y() = t.y() + u_bearing.y() * u_distance;
    v.z() = 0;
    
    if (*debug_count < 3) {
      // RCLCPP_INFO(get_logger(), "  result 3D: (%.3f, %.3f, %.3f)", v.x(), v.y(), v.z());
      (*debug_count)++;
    }
    
    return v;
  };
  return true;
}
void SegmentFilter::execute(const PointCloud2 & line_segments_msg, const Image & segment_msg)
{
  if (!define_project_func()) {
    using namespace std::literals::chrono_literals;
    RCLCPP_INFO_STREAM_THROTTLE(
      get_logger(), *get_clock(), (1000ms).count(), "project_func cannot be defined");
    return;
  }

  const rclcpp::Time stamp = line_segments_msg.header.stamp;

  pcl::PointCloud<pcl::PointNormal>::Ptr line_segments_cloud{
    new pcl::PointCloud<pcl::PointNormal>()};
  cv::Mat mask_image = common::decompress_to_cv_mat(segment_msg);
  pcl::fromROSMsg(line_segments_msg, *line_segments_cloud);

  // HJK: 디버그 - 입력 데이터 확인
  // RCLCPP_INFO(get_logger(), "Input line segments: %zu, mask size: %dx%d", 
  //             line_segments_cloud->size(), mask_image.cols, mask_image.rows);
  
  // if (!line_segments_cloud->empty()) {
  //   auto& first = line_segments_cloud->at(0);
  //   // RCLCPP_INFO(get_logger(), "First segment: (%.1f, %.1f) -> (%.1f, %.1f)",
  //   //             first.x, first.y, first.normal_x, first.normal_y);
  // }

  const std::set<int> indices = filter_by_mask(mask_image, *line_segments_cloud);
  
  // HJK: 디버그 - 필터링 결과
  // RCLCPP_INFO(get_logger(), "After filter_by_mask: %zu valid indices out of %zu total",
  //             indices.size(), line_segments_cloud->size());

  pcl::PointCloud<pcl::PointNormal> valid_edges = project_lines(*line_segments_cloud, indices);
  pcl::PointCloud<pcl::PointNormal> invalid_edges =
    project_lines(*line_segments_cloud, indices, true);

  // HJK: 디버그 - 투영 결과
  // RCLCPP_INFO(get_logger(), "After projection: %zu valid, %zu invalid",
  //             valid_edges.size(), invalid_edges.size());

  // Projected line segments
  {
    pcl::PointCloud<pcl::PointXYZLNormal> combined_edges;
    for (const auto & pn : valid_edges) {
      pcl::PointXYZLNormal pln;
      pln.getVector3fMap() = pn.getVector3fMap();
      pln.getNormalVector3fMap() = pn.getNormalVector3fMap();
      pln.label = 255;
      combined_edges.push_back(pln);
    }
    for (const auto & pn : invalid_edges) {
      pcl::PointXYZLNormal pln;
      pln.getVector3fMap() = pn.getVector3fMap();
      pln.getNormalVector3fMap() = pn.getNormalVector3fMap();
      pln.label = 0;
      combined_edges.push_back(pln);
    }
    common::publish_cloud(*pub_projected_cloud_, combined_edges, stamp);
  }

  // Image
  {
    cv::Mat projected_image = cv::Mat::zeros(cv::Size{image_size_, image_size_}, CV_8UC3);
    for (auto & pn : valid_edges) {
      cv::Point2i p1 = to_cv_point(pn.getVector3fMap());
      cv::Point2i p2 = to_cv_point(pn.getNormalVector3fMap());
      cv::line(projected_image, p1, p2, cv::Scalar(100, 100, 255), 4, cv::LineTypes::LINE_8);
    }
    for (auto & pn : invalid_edges) {
      cv::Point2i p1 = to_cv_point(pn.getVector3fMap());
      cv::Point2i p2 = to_cv_point(pn.getNormalVector3fMap());
      cv::line(projected_image, p1, p2, cv::Scalar(200, 200, 200), 3, cv::LineTypes::LINE_8);
    }
    common::publish_image(*pub_image_, projected_image, stamp);
  }

  // Line segments for debug
  {
    pcl::PointCloud<pcl::PointXYZLNormal> combined_debug_edges;
    for (size_t index = 0; index < line_segments_cloud->size(); ++index) {
      const pcl::PointNormal & pn = line_segments_cloud->at(index);
      pcl::PointXYZLNormal pln;
      pln.getVector3fMap() = pn.getVector3fMap();
      pln.getNormalVector3fMap() = pn.getNormalVector3fMap();
      if (indices.count(static_cast<int>(index)) > 0)
        pln.label = 255;
      else
        pln.label = 0;
      combined_debug_edges.push_back(pln);
    }
    common::publish_cloud(*pub_debug_cloud_, combined_debug_edges, stamp);
  }
}

bool SegmentFilter::is_near_element(
  const pcl::PointNormal & pn, pcl::PointNormal & truncated_pn) const
{
  float min_distance = std::min(pn.x, pn.normal_x);
  float max_distance = std::max(pn.x, pn.normal_x);
  if (min_distance > max_segment_distance_) return false;
  if (max_distance < max_segment_distance_) {
    truncated_pn = pn;
    return true;
  }

  truncated_pn = pn;
  Eigen::Vector3f t = pn.getVector3fMap() - pn.getNormalVector3fMap();
  float not_zero_tx = t.x() > 0 ? std::max(t.x(), 1e-3f) : std::min(t.x(), -1e-3f);
  float lambda = (max_segment_distance_ - pn.x) / not_zero_tx;
  Eigen::Vector3f m = pn.getVector3fMap() + lambda * t;
  if (pn.x > pn.normal_x)
    truncated_pn.getVector3fMap() = m;
  else
    truncated_pn.getNormalVector3fMap() = m;
  return true;
}

std::set<ushort> get_unique_pixel_value(cv::Mat & image)
{
  // `image` is a set of ushort.
  // The purpose is to find the unique set of values contained in `image`.
  // For example, if `image` is {0,1,2,0,1,2,3}, this function returns {0,1,2,3}.

  if (image.depth() != CV_16U) throw std::runtime_error("image's depth must be ushort");

  auto begin = image.begin<ushort>();
  auto last = std::unique(begin, image.end<ushort>());
  std::sort(begin, last);
  last = std::unique(begin, last);
  return {begin, last};
}

pcl::PointCloud<pcl::PointNormal> SegmentFilter::project_lines(
  const pcl::PointCloud<pcl::PointNormal> & points, const std::set<int> & indices,
  bool negative) const
{
  pcl::PointCloud<pcl::PointNormal> projected_points;
  
  int total_checked = 0;
  int failed_project1 = 0;
  int failed_project2 = 0;
  int failed_length = 0;
  int failed_lateral = 0;
  int failed_distance = 0;
  int success = 0;
  
  for (size_t index = 0; index < points.size(); ++index) {
    if (negative) {
      if (indices.count(static_cast<int>(index)) > 0) continue;
    } else {
      if (indices.count(static_cast<int>(index)) == 0) continue;
    }
    
    total_checked++;
    
    // HJK: 첫 번째 것만 상세 로그
    // if (total_checked == 1) {
    //   const auto& pn = points.at(index);
    //   // RCLCPP_INFO(get_logger(), "First checked segment pixel: (%.1f, %.1f) -> (%.1f, %.1f)",
    //   //             pn.x, pn.y, pn.normal_x, pn.normal_y);
    // }

    pcl::PointNormal truncated_pn = points.at(index);

    std::optional<Eigen::Vector3f> opt1 = project_func_(truncated_pn.getVector3fMap());
    std::optional<Eigen::Vector3f> opt2 = project_func_(truncated_pn.getNormalVector3fMap());
    
    if (!opt1.has_value()) {
      failed_project1++;
      if (total_checked == 1) {
        // RCLCPP_INFO(get_logger(), "First segment: projection 1 failed");
      }
      continue;
    }
    if (!opt2.has_value()) {
      failed_project2++;
      if (total_checked == 1) {
        // RCLCPP_INFO(get_logger(), "First segment: projection 2 failed");
      }
      continue;
    }
    
    // HJK: 첫 번째 투영 결과 출력
    if (total_checked == 1) {
      // RCLCPP_INFO(get_logger(), "First projected to 3D: (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)",
      //             opt1->x(), opt1->y(), opt1->z(), opt2->x(), opt2->y(), opt2->z());
    }

    // If line segment has shorter length than config, it is excluded
    if (min_segment_length_ > 0) {
      float length = (opt1.value() - opt2.value()).norm();
      if (length < min_segment_length_) {
        failed_length++;
        if (total_checked == 1) {
          // RCLCPP_INFO(get_logger(), "First segment: length %.3f < min %.3f", 
          //             length, min_segment_length_);
        }
        continue;
      }
    }
    
    if (max_lateral_distance_ > 0) {
      float abs_lateral1 = std::abs(opt1.value().y());
      float abs_lateral2 = std::abs(opt2.value().y());
      if (std::min(abs_lateral1, abs_lateral2) > max_lateral_distance_) {
        failed_lateral++;
        if (total_checked == 1) {
          // RCLCPP_INFO(get_logger(), "First segment: lateral %.3f > max %.3f",
          //             std::min(abs_lateral1, abs_lateral2), max_lateral_distance_);
        }
        continue;
      }
    }

    pcl::PointNormal xyz;
    xyz.x = opt1->x();
    xyz.y = opt1->y();
    xyz.z = opt1->z();
    xyz.normal_x = opt2->x();
    xyz.normal_y = opt2->y();
    xyz.normal_z = opt2->z();

    pcl::PointNormal truncated_xyz = xyz;
    if (max_segment_distance_ > 0) {
      if (!is_near_element(xyz, truncated_xyz)) {
        failed_distance++;
        if (total_checked == 1) {
          // RCLCPP_INFO(get_logger(), "First segment: distance check failed (max: %.2f)",
          //             max_segment_distance_);
        }
        continue;
      }
    }

    projected_points.push_back(truncated_xyz);
    success++;
  }
  
  // RCLCPP_INFO(get_logger(), "project_lines summary: total=%d, proj_fail=%d/%d, len_fail=%d, lat_fail=%d, dist_fail=%d, SUCCESS=%d",
  //             total_checked, failed_project1, failed_project2, 
  //             failed_length, failed_lateral, failed_distance, success);
  
  return projected_points;
}

std::set<int> SegmentFilter::filter_by_mask(
  const cv::Mat & mask, const pcl::PointCloud<pcl::PointNormal> & edges)
{
  // Create line image and assign different color to each segment.
  cv::Mat line_image = cv::Mat::zeros(mask.size(), CV_16UC1);
  for (size_t i = 0; i < edges.size(); i++) {
    auto & pn = edges.at(i);
    Eigen::Vector3f p1 = pn.getVector3fMap();
    Eigen::Vector3f p2 = pn.getNormalVector3fMap();
    cv::Scalar color = cv::Scalar::all(static_cast<double>(i + 1));
    cv::line(
      line_image, cv::Point2i(static_cast<int>(p1.x()), static_cast<int>(p1.y())),
      cv::Point2i(static_cast<int>(p2.x()), static_cast<int>(p2.y())), color, 1,
      cv::LineTypes::LINE_4);
  }

  cv::Mat mask_image;
  mask.convertTo(mask_image, CV_16UC1);
  cv::threshold(mask_image, mask_image, 1, std::numeric_limits<ushort>::max(), cv::THRESH_BINARY);

  // TODO(KYabuuchi) Using boost::geometry is more intuitive.
  // https://boostjp.github.io/tips/geometry.html#disjoint

  // And operator
  cv::Mat masked_line;
  cv::bitwise_and(mask_image, line_image, masked_line);
  std::set<ushort> pixel_values = get_unique_pixel_value(masked_line);

  // Extract edges within masks
  std::set<int> reliable_indices;
  for (size_t i = 0; i < edges.size(); i++) {
    if (pixel_values.count(i + 1) != 0) reliable_indices.insert(static_cast<int>(i));
  }

  return reliable_indices;
}

}  // namespace yabloc::segment_filter

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(yabloc::segment_filter::SegmentFilter)
