#ifndef MOTOR_CONTROLLER_HPP_
#define MOTOR_CONTROLLER_HPP_

#include <rclcpp/rclcpp.hpp>
#include "MotorManager.h"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

class MotorController : public rclcpp::Node
{
public:
    MotorController();

private:
    void update_motor();
    void publish_jointstate();
    void trajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);

    rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_subscriber_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_update;
    rclcpp::TimerBase::SharedPtr timer_publish;
    size_t count_;
    MotorManager motor_manager;
};

#endif  // MOTOR_CONTROLLER_HPP_