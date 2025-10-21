#!/usr/bin/env python3
import numpy as np
import sys
import coinespy as cpy
from bmi_common import (
    BMI08X,
    SensorType,
    BMI08X_16_BIT_RESOLUTION,
    BMI08X_ACCEL_ODR_200_HZ,
    BMI08X_ACCEL_BW_NORMAL,
    BMI08X_GYRO_BW_23_ODR_200_HZ,
    lsb_to_mps2,
    lsb_to_dps,
)

import time

def _s16(val: int) -> int:
    return val - 65536 if (val & 0x8000) else val


class imu_wrapper:
    def __init__(self):
        try:
            i2c_bus = list(cpy.I2CBus)[0]
        except Exception:
            print("No I2C bus available")
            sys.exit(1)

        self.imu = BMI08X(i2c_bus, cpy.SensorInterface.I2C)
        self.imu.init_board()

        # Power up sensors
        self.imu.set_accel_power_mode()
        self.imu.set_gyro_power_mode()

        # Configure ODR/bandwidth and keep current ranges from bmi_common defaults
        self.imu.accel_cfg["ODR"] = BMI08X_ACCEL_ODR_200_HZ
        self.imu.accel_cfg["BANDWIDTH"] = BMI08X_ACCEL_BW_NORMAL
        accel_range_reg = self.imu.set_accel_meas_conf()  # returns register value 0..3

        self.imu.gyro_cfg["ODR"] = BMI08X_GYRO_BW_23_ODR_200_HZ
        gyro_range_reg = None
        while gyro_range_reg is None:
            try:
                gyro_range_reg = self.imu.set_gyro_meas_conf()  # returns register value 0..4
            except:
                print("Error setting gyro measurement config, retrying...")
                time.sleep(0.5)

        # Map register range to physical range
        if getattr(self.imu, "imu_variant", "") == "BMI088":
            accel_g_map = {0: 3, 1: 6, 2: 12, 3: 24}
        else:
            accel_g_map = {0: 2, 1: 4, 2: 8, 3: 16}
        self.accel_g = accel_g_map.get(accel_range_reg & 0x03, 16)

        gyro_dps_map = {0: 2000, 1: 1000, 2: 500, 3: 250, 4: 125}
        self.gyro_dps = gyro_dps_map.get(gyro_range_reg & 0x07, 250)

    def read_accel_gyro(self):
        # Read accel (little-endian: LSB, MSB)
        ab = self.imu.read(SensorType.ACCEL, self.imu.ACCEL_DATA_ADDR, self.imu.ACCEL_DATA_REG_LEN)
        ax_lsb = _s16((ab[1] << 8) | ab[0])
        ay_lsb = _s16((ab[3] << 8) | ab[2])
        az_lsb = _s16((ab[5] << 8) | ab[4])
        ax = lsb_to_mps2(ax_lsb, self.accel_g, BMI08X_16_BIT_RESOLUTION)
        ay = lsb_to_mps2(ay_lsb, self.accel_g, BMI08X_16_BIT_RESOLUTION)
        az = lsb_to_mps2(az_lsb, self.accel_g, BMI08X_16_BIT_RESOLUTION)

        # Read gyro
        gb = self.imu.read(SensorType.GYRO, self.imu.GYRO_DATA_ADDR, self.imu.GYRO_DATA_REG_LEN)
        gx_lsb = _s16((gb[1] << 8) | gb[0])
        gy_lsb = _s16((gb[3] << 8) | gb[2])
        gz_lsb = _s16((gb[5] << 8) | gb[4])
        gx = lsb_to_dps(gx_lsb, self.gyro_dps, BMI08X_16_BIT_RESOLUTION)
        gy = lsb_to_dps(gy_lsb, self.gyro_dps, BMI08X_16_BIT_RESOLUTION)
        gz = lsb_to_dps(gz_lsb, self.gyro_dps, BMI08X_16_BIT_RESOLUTION)

        acc = np.array([ax, ay, az])
        angvel = np.array([gx, gy, gz]) * 2 * np.pi / 360

        return acc, angvel
    
if __name__ == "__main__":
    imu = imu_wrapper()
    print("Reading accel and gyro data:")
    while True:
        acc, angvel = imu.read_accel_gyro()
        print(f"Accel (m/s^2): {acc}, Gyro (rad/s): {angvel}")