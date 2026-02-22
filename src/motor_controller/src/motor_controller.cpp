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
    current_positions_.resize(12, 0.0);
    current_velocities_.resize(12, 0.0);

    publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
    timer_update = this->create_wall_timer(100us, std::bind(&MotorController::update_motor, this));
    timer_publish = this->create_wall_timer(500us, std::bind(&MotorController::publish_jointstate, this));
    trajectory_subscriber_ = this->create_subscription<gz_sim_interfaces::msg::MotorCmd>(
        "/real_joint_trajectories", 10,
        std::bind(&MotorController::trajectoryCallback, this, std::placeholders::_1)
    );
    float pelvis_offsets[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float left_offsets[6] = {0.0, 0.0, 0.0, 0.0, 0.0 , 0.0};
    float right_offsets[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    motor_manager.set_q_offsets(pelvis_offsets, left_offsets, right_offsets);
}

void MotorController::trajectoryCallback(const gz_sim_interfaces::msg::MotorCmd::SharedPtr msg)
{
    const std::map<std::string, std::pair<float, float>> joint_limits= {
        {"l_hip_pitch_joint", {-1.3, 1.3}},
        {"l_hip_roll_joint", {-0.4, 1.5}},
        {"l_hip_yaw_joint", {-1.0, 1.0}},
        {"l_knee_joint", {-0.3, 2.0}},
        {"l_foot_pitch_joint", {-2.0, 2.0}},
        {"l_foot_roll_joint", {-1.0, 1.0}},
        {"r_hip_pitch_joint", {-1.3, 1.3}},
        {"r_hip_roll_joint", {-1.5, 0.4}},
        {"r_hip_yaw_joint", {-1.0, 1.0}},
        {"r_knee_joint", {-0.3, 2.0}},
        {"r_foot_pitch_joint", {-2.0, 2.0}},
        {"r_foot_roll_joint", {-1.0, 1.0}}
    };

    const float SOFT_LIMIT_MARGIN = 0.174533f; // ~10 degrees
    const float VEL_SOFT_LIMIT = 10.0f;
    const float VEL_HARD_LIMIT = 15.0f;

    std::stringstream ss;
    for (int i = 0; i < 12; i++) {
        ss << msg->positions[i] << " ";
    }
    //RCLCPP_INFO(this->get_logger(), "Trajectory positions: %s", ss.str().c_str());

    float shutdown_fac = 1.0;

    for (int i = 0; i < (int)msg->joint_names.size(); i++) {
        const std::string& joint_name = msg->joint_names[i];
        auto it = joint_limits.find(joint_name);
        if (it != joint_limits.end()) {
            double real_pos = current_positions_[i];
            double real_vel = current_velocities_[i];
            auto lims = it->second;
            if (real_pos <= lims.first || 
                real_pos >= lims.second ||
                std::abs(real_vel) >= VEL_HARD_LIMIT) {
                    shutdown_fac = 0.0;
                    RCLCPP_INFO(this->get_logger(), "Hard Limit Exceeded");
            } else if(real_pos <= lims.first + SOFT_LIMIT_MARGIN ||
                      real_pos >= lims.second - SOFT_LIMIT_MARGIN ||
                      std::abs(real_vel) >= VEL_SOFT_LIMIT) {
                msg->kp[i] /= 2.0;
                msg->kd[i] /= 2.0;
            }
        }
    }

    // Log non-zero motor error codes
    {
        bool any_error = false;
        std::ostringstream oss;
        oss << "Motor error codes:";
        for (int i = 0; i < 18; ++i) {
            int code = motor_manager.error_codes[i];
            if (code != 0) {
                any_error = true;
                oss << " [" << i << "]=" << code;
            }
        }
        if (any_error) {
            RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
        }
    }

    shutdown_fac = 1.0;

    //int double_arr[6] = {0, 3, 4, 6, 9, 10};

    for (int i = 0; i < 12; i++) {
        if (i == 0 || i == 3 || i == 4 || i == 6 || i == 9 || i == 10) {
            motor_manager.joint_state[i].kp = msg->kp[i] * shutdown_fac * 0.5;
            motor_manager.joint_state[i].kd = msg->kd[i] * shutdown_fac * 0.5;
            motor_manager.joint_state[i].tau = msg->torques[i] * shutdown_fac * 0.5;
        }
        else {
            motor_manager.joint_state[i].kp = msg->kp[i] * shutdown_fac;
            motor_manager.joint_state[i].kd = msg->kd[i] * shutdown_fac;
            motor_manager.joint_state[i].tau = msg->torques[i] * shutdown_fac;
        }
        motor_manager.joint_state[i].des_p = msg->positions[i];
        motor_manager.joint_state[i].des_d = msg->velocities[i];

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
    message.name.resize(18);
    message.position.resize(18);
    message.velocity.resize(18);
    message.effort.resize(18);

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

    message.name[12] = "l_shoulder_pitch_joint";
    message.position[12] = 0.0;
    message.velocity[12] = 0.0;

    message.name[13] = "l_shoulder_roll_joint";
    message.position[13] = 0.05;
    message.velocity[13] = 0.0;

    message.name[14] = "l_elbow_joint";
    message.position[14] = 0.0;
    message.velocity[14] = 0.0;

    message.name[15] = "r_shoulder_pitch_joint";
    message.position[15] = 0.0;
    message.velocity[15] = 0.0;

    message.name[16] = "r_shoulder_roll_joint";
    message.position[16] = -0.05;
    message.velocity[16] = 0.0;

    message.name[17] = "r_elbow_joint";
    message.position[17] = 0.0;
    message.velocity[17] = 0.0;

    for (int i = 0; i < 12; i++) {
        current_positions_[i] = message.position[i];
        current_velocities_[i] = message.velocity[i];
    }

    publisher_->publish(message);

}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotorController>());
    rclcpp::shutdown();
    return 0;
}