cmake_minimum_required(VERSION 3.8)
project(motor_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(include)
include_directories(include/)
link_directories(lib2)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)


if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64") 
    set(EXTRA_LIBS libUnitreeMotorSDK_Arm64.so)
else()
    set(EXTRA_LIBS libUnitreeMotorSDK_Linux64.so)
endif()

add_library(motor_manager src/MotorManager.cpp)
add_executable(motor_controller src/motor_controller.cpp)
target_link_libraries(motor_controller
  motor_manager
  ${EXTRA_LIBS}
)
ament_target_dependencies(motor_controller
  rclcpp
  sensor_msgs
  trajectory_msgs
)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

set_target_properties(motor_controller PROPERTIES INSTALL_RPATH "$ORIGIN/..")

install(TARGETS
  motor_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES ${CMAKE_SOURCE_DIR}/lib2/${EXTRA_LIBS}
  DESTINATION lib
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
