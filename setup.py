from setuptools import find_packages, setup
import os
from glob import glob

package_name = "physics_atv_visual_mapping"
data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    (os.path.join("share", package_name), glob("launch/*.py")),
    (os.path.join("share", package_name, "config", "ros"), glob("config/ros/*.yaml")),
]

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cherieh",
    maintainer_email="cherieh@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dino_localmapping = physics_atv_visual_mapping.dino_localmapping:main",
            "dino_cost = physics_atv_visual_mapping.dino_cost:main",
            "odom_to_tf = physics_atv_visual_mapping.odom_to_tf:main",
        ],
    },
)
