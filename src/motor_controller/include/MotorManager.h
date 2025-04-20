#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"
#include <string>
#include <memory>

struct RawMotorStruct {
    float kp = 0.0;
    float kd = 0.0;
    float des_p = 0.0;
    float des_d = 0.0;
    float tau = 0.0;
};

struct JointStateStruct {
    float kp;
    float kd;
    float des_p;
    float des_d;
    float tau;
    float current_q = 0.0;
    float current_dq = 0.0;
};

class SingleMotorManager {
    private:
        MotorCmd cmd[6];
        MotorData data[6];
        std::unique_ptr<SerialPort> serial;
        bool serial_init = false;
        float find_q(float cur_q, float des_q);
    public:
        struct RawMotorStruct raw_motor[6];
        float q_offsets[6];
        float raw_q_motor[6];
        float raw_dq_motor[6];
        int motor_error[6];
        SingleMotorManager(std::string port = "/dev/ttyUSB0");
        ~SingleMotorManager();
        void update();
        void printMotorData();
        void set_q_offsets(float q[6]);
};

class MotorManager {
    private:
        void assignMotorCmd(struct JointStateStruct &data, struct RawMotorStruct &raw, float mult);
        // 0: l_hip_pitch
        // 1: l_hip_roll
        // 2: l_hip_yaw
        // 3: l_knee
        // 4: l_ankle_pitch
        // 5: l_ankle_roll
        // 6: r_hip_pitch
        // 7: r_hip_roll
        // 8: r_hip_yaw
        // 9: r_knee
        // 10: r_ankle_pitch
        // 11: r_ankle_roll
        SingleMotorManager pelvis;
        SingleMotorManager left;
        SingleMotorManager right;
        bool safe;
        float gear_ratio;
    public:
        void set_q_offsets(float pelvis_des_q[6], float left_des_q[6], float right_des_q[6]);
        struct JointStateStruct joint_state[12];
        MotorManager();
        ~MotorManager();
        void update();
};