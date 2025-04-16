#include <filesystem>
#include <unistd.h>
#include "MotorManager.h"
#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"
#include <string>
#include <stdexcept>

SingleMotorManager::SingleMotorManager(std::string port){
    try {
        serial = std::make_unique<SerialPort>(port);
        serial_init = true;
    } catch (const std::exception &e) {
        std::cout << "No usb found on port " << port << std::endl;
    }
    for (int i = 0; i < 6; i++) {
        motor_error[i] = -1;
        raw_q_motor[i] = 0.0;
        raw_dq_motor[i] = 0.0;
        cmd[i].motorType = MotorType::GO_M8010_6;
        data[i].motorType = MotorType::GO_M8010_6;
        cmd[i].mode = queryMotorMode(MotorType::GO_M8010_6,MotorMode::FOC);
        cmd[i].id   = i;
        cmd[i].kp   = 0.0;
        cmd[i].kd   = 0.0;
        cmd[i].tau = 0.0;
    }
}

void SingleMotorManager::update() {
    for (int i = 0; i < 6; i++) {
        if (motor_error[i] == -1) {
            cmd[i].kp = 0.0;
            cmd[i].kd = 0.0;
            cmd[i].tau = 0.0;
        } else {
            cmd[i].kp = raw_motor[i].kp;
            cmd[i].kd = raw_motor[i].kd;
            cmd[i].q = raw_motor[i].des_p;
            cmd[i].dq = raw_motor[i].des_d;
            cmd[i].tau = raw_motor[i].tau;
        }
        if (serial_init) {
            serial -> sendRecv(&cmd[i], &data[i]);
        }
        raw_q_motor[i] = data[i].q;
        raw_dq_motor[i] = data[i].dq;
        motor_error[i] = 0;
        if (data[i].merror != 0) {
            motor_error[i] = -1;
        }
        if (data[i].temp >= 100) {
            motor_error[i] = -1;
        }   
    }
}

void SingleMotorManager::printMotorData() {
    std::cout << "==================================================" << std::endl;
    for (int i = 0; i < 6; i++) {
        std::cout << "Motor " << i << ": " << std::endl;
        std::cout << "  q: " << data[i].q << std::endl;
        std::cout << "  dq: " << data[i].dq << std::endl;
        std::cout << "  temp: " << data[i].temp << std::endl;
        std::cout << "  merror: " << data[i].merror << std::endl;
    }
}

SingleMotorManager::~SingleMotorManager() {
}

MotorManager::MotorManager() :
    pelvis("/dev/ttyUSB0"),
    left("/dev/ttyUSB1"),
    right("/dev/ttyUSB2")
{
    gear_ratio = queryGearRatio(MotorType::GO_M8010_6);
    safe = false;
    // unsafe means that reading is still engaged but no motor cmds will be sent

}

MotorManager::~MotorManager() {
}

void MotorManager::assignMotorCmd(struct JointStateStruct &data, struct RawMotorStruct &raw, float mult) {
    double test_tau = data.tau + data.kp * (data.des_p - data.current_q) + data.kd * (data.des_d - data.current_dq);
    double kp = data.kp / (gear_ratio * gear_ratio);
    double kd = data.kd / (gear_ratio * gear_ratio);
    if (abs(test_tau) > 23) {
        kp = kp * 23 / abs(test_tau);
        kd = kd * 23 / abs(test_tau);
    }
    raw.kp = data.kp / (gear_ratio * gear_ratio);
    raw.kd = data.kd / (gear_ratio * gear_ratio);
    raw.des_p = data.des_p * mult * gear_ratio;
    raw.des_d = data.des_d * mult * gear_ratio;
    raw.tau = data.tau * mult * gear_ratio;
}

void MotorManager::update() {

    // Go through each joint and set the commands for each controller

    if (safe) {
        // 0: l_hip_pitch
        assignMotorCmd(joint_state[0], pelvis.raw_motor[2], 1.0);
        assignMotorCmd(joint_state[0], left.raw_motor[1], -1.0);

        // 1: l_hip_roll
        assignMotorCmd(joint_state[1], pelvis.raw_motor[0], -1.0);

        // 2: l_hip_yaw
        assignMotorCmd(joint_state[2], left.raw_motor[0], -1.0);

        // 3: l_knee
        assignMotorCmd(joint_state[3], left.raw_motor[1], 1.0);
        assignMotorCmd(joint_state[3], left.raw_motor[2], -1.0);

        // 4: l_ankle_pitch
        assignMotorCmd(joint_state[4], left.raw_motor[3], -1.0);
        assignMotorCmd(joint_state[4], left.raw_motor[4], 1.0);

        // 5: l_ankle_roll
        assignMotorCmd(joint_state[5], left.raw_motor[5], 1.0);

        // 6: r_hip_pitch
        assignMotorCmd(joint_state[6], pelvis.raw_motor[4], 1.0);
        assignMotorCmd(joint_state[6], pelvis.raw_motor[5], -1.0);

        // 7: r_hip_roll
        assignMotorCmd(joint_state[7], pelvis.raw_motor[3], -1.0);

        // 8: r_hip_yaw
        assignMotorCmd(joint_state[8], right.raw_motor[0], -1.0);

        // 9: r_knee
        assignMotorCmd(joint_state[9], right.raw_motor[1], 1.0);
        assignMotorCmd(joint_state[9], right.raw_motor[2], -1.0);

        // 10: r_ankle_pitch
        assignMotorCmd(joint_state[10], right.raw_motor[3], -1.0);
        assignMotorCmd(joint_state[10], right.raw_motor[4], 1.0);

        // 11: r_ankle_roll
        assignMotorCmd(joint_state[11], right.raw_motor[5], 1.0);
    }


    left.update();
    pelvis.update();
    right.update();
    safe = true;
    for (int i = 0; i < 6; i++) {
        if (left.motor_error[i] == -1 || pelvis.motor_error[i] == -1 || right.motor_error[i] == -1) {
            safe = false;
        }
    }
    // Go through each of the 18 joints and update the joint state

    // 0: l_hip_pitch
    joint_state[0].current_q = (
        left.raw_q_motor[2] + left.raw_q_motor[1] * -1) / (2 * gear_ratio);
    joint_state[0].current_dq = (
        left.raw_dq_motor[2] + left.raw_dq_motor[1] * -1) / (2 * gear_ratio);

    // 1: l_hip_roll
    joint_state[1].current_q = pelvis.raw_q_motor[0] * -1 / gear_ratio;
    joint_state[1].current_dq = pelvis.raw_dq_motor[0] * -1 / gear_ratio;

    // 2: l_hip_yaw
    joint_state[2].current_q = left.raw_q_motor[0] * -1 / gear_ratio;
    joint_state[2].current_dq = left.raw_dq_motor[0] * -1 / gear_ratio;

    // 3: l_knee
    joint_state[3].current_q = (
        left.raw_q_motor[1] + left.raw_q_motor[2] * -1) / (2 * gear_ratio);
    joint_state[3].current_dq = (
        left.raw_dq_motor[1] + left.raw_dq_motor[2] * -1) / (2 * gear_ratio);
    
    // 4: l_ankle_pitch

    joint_state[4].current_q = (
        left.raw_q_motor[4] + left.raw_q_motor[3] * -1) / (2 * gear_ratio);
    joint_state[4].current_dq = (
        left.raw_dq_motor[4] + left.raw_dq_motor[3] * -1) / (2 * gear_ratio);

    // 5: l_ankle_roll

    joint_state[5].current_q = left.raw_q_motor[5] / gear_ratio;
    joint_state[5].current_dq = left.raw_dq_motor[5] / gear_ratio;

    // 6: r_hip_pitch

    joint_state[6].current_q = (
        pelvis.raw_q_motor[4] + pelvis.raw_q_motor[5] * -1) / (2 * gear_ratio);
    joint_state[6].current_dq = (
        pelvis.raw_dq_motor[4] + pelvis.raw_dq_motor[5] * -1) / (2 * gear_ratio);

    // 7: r_hip_roll

    joint_state[7].current_q = pelvis.raw_q_motor[3] * -1 / gear_ratio;
    joint_state[7].current_dq = pelvis.raw_dq_motor[3] * -1 / gear_ratio;

    // 8: r_hip_yaw
    
    joint_state[8].current_q = right.raw_q_motor[0] * -1 / gear_ratio;
    joint_state[8].current_dq = right.raw_dq_motor[0] * -1 / gear_ratio;

    // 9: r_knee
    joint_state[9].current_q = (
        right.raw_q_motor[2] + right.raw_q_motor[1] * -1) / (2 * gear_ratio);
    joint_state[9].current_dq = (
        right.raw_dq_motor[2] + right.raw_dq_motor[1] * -1) / (2 * gear_ratio);

    // 10: r_ankle_pitch
    joint_state[10].current_q = (
        right.raw_q_motor[3] + right.raw_q_motor[4] * -1) / (2 * gear_ratio);
    joint_state[10].current_dq = (
        right.raw_dq_motor[3] + right.raw_dq_motor[4] * -1) / (2 * gear_ratio);

    // 11: r_ankle_roll
    joint_state[11].current_q = right.raw_q_motor[5] / gear_ratio;
    joint_state[11].current_dq = right.raw_dq_motor[5] / gear_ratio;
}