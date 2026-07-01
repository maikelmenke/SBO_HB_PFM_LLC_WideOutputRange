from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys

# Read long description from README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


class PostInstallCommand(install):
    """Post-installation hook: attempts to install the ngspice shared library.

    Runs ``pyspice-post-installation --install-ngspice-dll`` after the normal
    ``pip install`` completes.  Failure is non-fatal: the user is shown the
    manual command to run instead.
    """

    def run(self):
        install.run(self)
        try:
            # PySpice registers 'pyspice-post-installation' as a console
            # script. Invoking via the full module path makes it work even
            # when the Scripts/bin folder is not yet on PATH.
            subprocess.check_call([
                sys.executable,
                "-m", "PySpice.Scripts.pyspice_post_installation",
                "--install-ngspice-dll",
            ])
            print("ngspice DLL successfully installed.")
        except Exception as e:
            print("Failed to install ngspice DLL automatically.")
            print("    Run manually: pyspice-post-installation --install-ngspice-dll")
            print("    Error:", e)


setup(
    name="llc_simulator",
    version="0.1.0",
    description="LLC converter simulator using PySpice/ngspice",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GSEC",
    license="GPL-3.0-or-later",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "examples*"]),
    include_package_data=True,
    package_data={
        "llc_devices": ["db/*.json", "db/*.ndjson"],
    },
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "seaborn>=0.13",
        "tabulate>=0.9",
        "PySpice>=1.5",
        "pymoo>=0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov",
            "ruff",
            "mypy",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
    cmdclass={
        "install": PostInstallCommand,
    },
)