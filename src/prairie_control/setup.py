from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'prairie_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/meshes/nemo4b', glob('meshes/nemo4b/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/helpers', glob('helpers/*')),
        ('share/' + package_name + '/walk_policy', glob('walk_policy/*')),
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
            'home_pd = prairie_control.home_pd:main'
        ],
    },
)
