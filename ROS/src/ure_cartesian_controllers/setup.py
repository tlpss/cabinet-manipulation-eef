import os
from glob import glob

from setuptools import setup

package_name = "ure_cartesian_controllers"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # include all launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "urdf"),glob("urdf/*.xacro") ),
        (os.path.join("share", package_name, "urdf/inc"),glob("urdf/inc/*.xacro") ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="tlips",
    maintainer_email="thomas17.lips@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    entry_points={
        # these make the modules executable from the command line by running ros2 run <package_name> <script_name>
        # otherwise you have to run python <path_to_script>
        "console_scripts": ["tf2_to_topic_publisher = ure_cartesian_controllers.tf2_to_topic_publisher:main"],
    },
)
