from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'gz_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', 
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/meshes/nemo', glob('meshes/nemo/*')),
        ('share/' + package_name + '/meshes/nemo3', glob('meshes/nemo3/*')),
        ('share/' + package_name + '/meshes/g1', glob('meshes/g1/*')),
        ('share/' + package_name + '/sdf', glob('sdf/*')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/helpers', glob('helpers/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jshiao',
    maintainer_email='jshiao@purdue.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gz_state_observer = gz_sim.gz_state_observer:main'
        ],
    },
)
