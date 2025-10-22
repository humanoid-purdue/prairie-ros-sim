#include <filesystem>
#include <unistd.h>
#include "MotorManager.h"
#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"
#include <string>
#include <stdexcept>
#include <cmath>

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
        q_offsets[i] = 0.0;
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
            cmd[i].q = raw_motor[i].des_p - q_offsets[i];
            cmd[i].dq = raw_motor[i].des_d;
            cmd[i].tau = raw_motor[i].tau;
        }
        if (serial_init) {
            serial -> sendRecv(&cmd[i], &data[i]);
        }
        raw_q_motor[i] = data[i].q + q_offsets[i];
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

float SingleMotorManager::find_q(float cur_q, float des_q) {
    float min_sep = 10.0;
    float min_offset = 0.0;
    for (int i = -6; i < 7; i++) {
        for (int j = -5; j < 6; j++) {
            float probe_q = std::fmod((6.33 * M_PI / 6) * i, 2.0f * M_PI) + j * 2 * M_PI;
            //float probe_q = 6.33 * M_PI / 3;
            float probe_sep = std::abs(cur_q + probe_q - des_q);
            if (probe_sep < min_sep) {
                min_sep = probe_sep;
                min_offset = probe_q;
            }
        }
    }
    return min_offset;
}

// float SingleMotorManager::find_q(float cur_q, float des_q) {
//     float min_sep = 10.0;
//     float min_offset = 0.0;
//     for (int i = -6; i <= 6; i++) {
//         float full_offset = i * (M_PI / 3.0f);
        
//         for (int j = -6; j < 6; j++) {
//             float inner_offset = (6.33f * M_PI / 3.0f) * j;
//             float total_offset = std::fmod(inner_offset + full_offset, M_PI * 2.0f);

//             float probe_sep = std::abs(cur_q + total_offset - des_q);
//             if (probe_sep < min_sep) {
//                 min_sep = probe_sep;
//                 min_offset = total_offset;
//             }
//         }
            

//     }
//     return min_offset;
// }

void SingleMotorManager::set_q_offsets(float q[6]) {
    std::cout << "------------------------" << std::endl;
    for (int i = 0; i < 6; i++) {
        q_offsets[i] = find_q(raw_q_motor[i], q[i]);
        std::cout << raw_q_motor[i] << " | " << q[i] << " | " << q_offsets[i] << std::endl;
    }
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
    raw.kp = kp;
    raw.kd = kd;
    raw.des_p = data.des_p * mult * gear_ratio;
    raw.des_d = data.des_d * mult * gear_ratio;
    raw.tau = data.tau * mult * gear_ratio;
}

void MotorManager::set_q_offsets(float pelvis_dq[6], float left_dq[6], float right_dq[6]) {
    safe = false;
    update();
    // To determine what offsets should be,
    // Go through each joint and determine what the approximate
    // q in motor q space is and find minimum pi based offset for that.
    // The motor q space would be in increments of 6.33 * p/3 radians modulo 2 pi
    pelvis.set_q_offsets(pelvis_dq);
    left.set_q_offsets(left_dq);
    right.set_q_offsets(right_dq);
}

void MotorManager::update() {

    // Go through each joint and set the commands for each controller

    if (safe) {
        // 0: l_hip_pitch
        assignMotorCmd(joint_state[0], pelvis.raw_motor[2], 1.0);
        assignMotorCmd(joint_state[0], pelvis.raw_motor[1], -1.0);

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
        assignMotorCmd(joint_state[9], right.raw_motor[1], -1.0);
        assignMotorCmd(joint_state[9], right.raw_motor[2], 1.0);

        // 10: r_ankle_pitch
        assignMotorCmd(joint_state[10], right.raw_motor[3], 1.0);
        assignMotorCmd(joint_state[10], right.raw_motor[4], -1.0);

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
        pelvis.raw_q_motor[2] + pelvis.raw_q_motor[1] * -1) / (2 * gear_ratio);
    joint_state[0].current_dq = (
        pelvis.raw_dq_motor[2] + pelvis.raw_dq_motor[1] * -1) / (2 * gear_ratio);

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