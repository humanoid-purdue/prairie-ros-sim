#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <sstream>

#include "motor_controller/MotorController.h"
#include "sensor_msgs/msg/joint_state.hpp"

using namespace std::chrono_literals;


MotorController::MotorController() : rclcpp::Node("motor_controller"), count_(0), motor_manager()
{
    publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    timer_update = this->create_wall_timer(100us, std::bind(&MotorController::update_motor, this));
    timer_publish = this->create_wall_timer(500us, std::bind(&MotorController::publish_jointstate, this));
    trajectory_subscriber_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
        "joint_trajectories", 10,
        std::bind(&MotorController::trajectoryCallback, this, std::placeholders::_1)
    );
    float pelvis_offsets[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float left_offsets[6] = {0.0, 0.0, 0.0, 0.0, 0.0 , 0.0};
    float right_offsets[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    motor_manager.set_q_offsets(pelvis_offsets, left_offsets, right_offsets);
}

void MotorController::trajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
    std::stringstream ss;
    for (int i = 0; i < 12; i++) {
        ss << msg->points[0].positions[i] << " ";
    }
    //RCLCPP_INFO(this->get_logger(), "Trajectory positions: %s", ss.str().c_str());
    float kp = 24.0;
    float kd = 1.0;
    bool all_zero = true;
    for (int i = 0; i < 12; i++) {
        if (msg ->points[0].positions[i] != 0.0 || msg->points[0].velocities[i] != 0.0) {
            all_zero = false;
        }
    }
    if (all_zero) {
        kp = 0.0;
        kd = 0.0;
    }
    for (int i = 0; i < 12; i++) {
        motor_manager.joint_state[i].kp = kp;
        motor_manager.joint_state[i].kd = kd;
        //if (i == 2 && !all_zero) {
        //    motor_manager.joint_state[i].kp = 4.0;
        //}
        motor_manager.joint_state[i].des_p = msg->points[0].positions[i];
        motor_manager.joint_state[i].des_d = msg->points[0].velocities[i];
    }
}

void MotorController::update_motor()
{
    motor_manager.update();
}

void MotorController::publish_jointstate()
{
    auto message = sensor_msgs::msg::JointState();
    message.header.stamp = this->now();
    message.name.resize(12);
    message.position.resize(12);
    message.velocity.resize(12);
    message.effort.resize(12);

    message.name[0] = "l_hip_pitch_joint";
    message.position[0] = motor_manager.joint_state[0].current_q;
    message.velocity[0] = motor_manager.joint_state[0].current_dq;

    message.name[1] = "l_hip_roll_joint";
    message.position[1] = motor_manager.joint_state[1].current_q;
    message.velocity[1] = motor_manager.joint_state[1].current_dq;

    message.name[2] = "l_hip_yaw_joint";
    message.position[2] = motor_manager.joint_state[2].current_q;
    message.velocity[2] = motor_manager.joint_state[2].current_dq;

    message.name[3] = "l_knee_joint";
    message.position[3] = motor_manager.joint_state[3].current_q;
    message.velocity[3] = motor_manager.joint_state[3].current_dq;

    message.name[4] = "l_foot_pitch_joint";
    message.position[4] = motor_manager.joint_state[4].current_q;
    message.velocity[4] = motor_manager.joint_state[4].current_dq;

    message.name[5] = "l_foot_roll_joint";
    message.position[5] = motor_manager.joint_state[5].current_q;
    message.velocity[5] = motor_manager.joint_state[5].current_dq;

    message.name[6] = "r_hip_pitch_joint";
    message.position[6] = motor_manager.joint_state[6].current_q;
    message.velocity[6] = motor_manager.joint_state[6].current_dq;

    message.name[7] = "r_hip_roll_joint";
    message.position[7] = motor_manager.joint_state[7].current_q;
    message.velocity[7] = motor_manager.joint_state[7].current_dq;

    message.name[8] = "r_hip_yaw_joint";
    message.position[8] = motor_manager.joint_state[8].current_q;
    message.velocity[8] = motor_manager.joint_state[8].current_dq;

    message.name[9] = "r_knee_joint";
    message.position[9] = motor_manager.joint_state[9].current_q;
    message.velocity[9] = motor_manager.joint_state[9].current_dq;

    message.name[10] = "r_foot_pitch_joint";
    message.position[10] = motor_manager.joint_state[10].current_q;
    message.velocity[10] = motor_manager.joint_state[10].current_dq;

    message.name[11] = "r_foot_roll_joint";
    message.position[11] = motor_manager.joint_state[11].current_q;
    message.velocity[11] = motor_manager.joint_state[11].current_dq;

    publisher_->publish(message);

}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotorController>());
    rclcpp::shutdown();
    return 0;
}