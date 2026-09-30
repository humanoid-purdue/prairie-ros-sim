from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'prairie_control'


def files_only(pattern):
    return [path for path in glob(pattern) if os.path.isfile(path)]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', files_only('urdf/*')),
        ('share/' + package_name + '/meshes/nemo4b', glob('meshes/nemo4b/*')),
        ('share/' + package_name + '/meshes/nemo6', glob('meshes/nemo6/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/helpers', glob('helpers/*')),
        ('share/' + package_name + '/walk_policy', glob('walk_policy/*')),
        ('share/' + package_name + '/data', glob('data/*')),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aurum',
    maintainer_email='ludwigtaycheeying@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'default_state = prairie_control.default_state:main',
            'default_pd = prairie_control.default_pd:main',
            'home_pd = prairie_control.home_pd:main',
            'static_pd = prairie_control.static_pd:main',
            'echo_keyboard = prairie_control.echo_keyboard:main',
            'gz_standing = prairie_control.gz_standing:main',
            'real_standing = prairie_control.real_standing:main',
            'gz_policy = prairie_control.gz_policy:main',
            'gz_mirror = prairie_control.gz_mirror:main',
            'real_policy = prairie_control.real_policy:main',
            'real_imu = prairie_control.real_imu:main',
            'real_state_estimator = prairie_control.real_state_estimator:main',
            'prairie_teleop = prairie_control.prairie_teleop:main',
            'prairie_keyboard_teleop = '
            'prairie_control.prairie_keyboard_teleop:main',
            'prairie_supervisor = prairie_control.prairie_supervisor:main',
            'prairie_command_mux = prairie_control.prairie_command_mux:main',
            'master = prairie_control.master:main',
            'master_test = prairie_control.master_test:main',
        ],
    },
)
