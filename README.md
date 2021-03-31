# Robotics Workshop #

This repository contains the installation instructions for required software and scripts of the workshop.

The following steps show how to install each component. Please contact the intructor team in the case that you encountered any problem.

## Python 3.7

If you have already installed Python 3.7, you may skip this part.

1. Open a terminal and start by updating the packages list and installing the prerequisites:

```shell
sudo apt update
sudo apt install software-properties-common
```

2. Next, add the deadsnakes PPA to your sources list:

```shell
sudo add-apt-repository ppa:deadsnakes/ppa
```

When prompted press `Enter` to continue.

3. Once the repository is enabled, install Python 3.7 with:

```shell
sudo apt install python3.7
```

4. At this point, Python 3.7 is installed on your Ubuntu system and ready to be used. You can verify it by typing:

```shell
python3.7 --version
```

## PyCharm

PyCharm is an *Integrated Development Environmnet* (IDE) used for programming in Python.

1. Go to the PyCharm [download page](https://www.jetbrains.com/pycharm/download/).
2. Select the proper operating system and choose the **Community** version.
3. Download the *.zip* file and unpack it. You will find a folder named `pycharm-community-2020.X.X`.
4. In order to start PyCharm, follow the below steps:
  * Open a terminal.
  * Change directory to `pycharm-community-2020.X.X/bin`.
  * Run `./pycharm.sh` in your termial.

## Pipenv

Pipenv is a tool that automatically creates and manages a virtualenv for your projects, as well as adds/removes packages from your `Pipfile` as you install/uninstall packages. In simple terms, pipenv helps you to install dependencies for your project without overwriting packages from other projects.

In order to install pipenv, type the following in your terminal:

```shell
pip install pipenv --user
```

## Installing Python Packages

```shell
git clone https://github.com/ExistentialRobotics/robotics-workshop.git
cd robotics-workshop
pipenv install --skip-lock
pipenv shell
```














