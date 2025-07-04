// Copyright 2025 TIER IV, Inc.
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

#ifndef TRAFFIC_LIGHT_HPP_
#define TRAFFIC_LIGHT_HPP_

#include <rclcpp/rclcpp.hpp>

#include <autoware_perception_msgs/msg/traffic_light_group.hpp>
#include <tier4_external_api_msgs/msg/traffic_light_group.hpp>

namespace tier4_autoware_api_extension
{

using ExternalMessage = tier4_external_api_msgs::msg::TrafficLightGroup;
using InternalMessage = autoware_perception_msgs::msg::TrafficLightGroup;

class TrafficLight : public rclcpp::Node
{
public:
  explicit TrafficLight(const rclcpp::NodeOptions & options);

private:
  void on_message(const InternalMessage & internal);
  rclcpp::Subscription<InternalMessage>::SharedPtr sub_traffic_light_group_;
  rclcpp::Publisher<ExternalMessage>::SharedPtr pub_traffic_light_group_;
};

}  // namespace tier4_autoware_api_extension

#endif  // TRAFFIC_LIGHT_HPP_
