# EAGLE Codebase

This repository implements **Adaptive SDN Autoscaling via Generalizable Multi-Agent Reinforcement Learning with EAGLE**: a decentralized learning framework designed to autonomously add or remove supporting switches and links to address network congestion.

---

## Table of Contents

- [EAGLE Codebase](#eagle-codebase)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Directory Structure](#directory-structure)
    - [`ding/`](#ding)
    - [`environment/`](#environment)
    - [`partitioning/`](#partitioning)
    - [`mininet_nets/`](#mininet_nets)
    - [`model/`](#model)
    - [`ryu_controller/`](#ryu_controller)
    - [`saved_models/`](#saved_models)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Mininet Topology Setup](#mininet-topology-setup)
    - [Ryu Controller Setup](#ryu-controller-setup)
      - [Install Requirements](#install-requirements)
      - [Start Controllers](#start-controllers)
      - [Stop Controllers](#stop-controllers)
    - [Reinforcement Learning (RL) Model Setup](#reinforcement-learning-rl-model-setup)
      - [Create Python Virtual Environment](#create-python-virtual-environment)
      - [Activate the Environment](#activate-the-environment)
      - [Install Python Dependencies](#install-python-dependencies)
      - [Train the RL Model](#train-the-rl-model)
      - [Test the RL Model](#test-the-rl-model)
    - [Traffic trace execution](#traffic-trace-execution)

---

## Project Overview

This repository provides the configuration and implementation for emulating adaptive SDN autoscaling using Mininet and Ryu. The framework leverages reinforcement learning to dynamically manage network resources, ensuring efficient handling of congestion in a decentralized manner.

---

## Directory Structure

Below is an overview of the key directories in this repository:

### `ding/`

This directory is part of an external open-source framework, **DI-engine**, used to implement reinforcement learning (RL) algorithms. It is included as a dependency for RL-related functionality. Within this directory, the `policy/decentralized.py` includes our custom policy implementation and `model/template/dec_mavac.py` the neural network definition. This module enables a multi-agent learning setup where individual agents independently learn strategies while coordinating collaboratively to optimize overall network performance.

For more information about DI-engine, visit the [DI-engine GitHub repository](https://github.com/opendilab/DI-engine).

### `environment/`

Contains modules for managing distributed environments and configurations. The environment collects observations from the controller, sends them to the RL model, receives actions from the model, and translates them into network configurations.

### `partitioning/`

Provides tools for graph partitioning and community detection.

### `mininet_nets/`

Contains topology configuration and traffic-related files for Mininet.

### `model/`

Contains scripts for training and testing decentralized reinforcement learning agents.

### `ryu_controller/`

Contains the implementation of an SDN controller that collects statistics from switches and communicates with the environment.

### `saved_models/`

Contains pre-trained model weights ready for testing.

---

## Installation

This project requires **Mininet** and **Ryu** installed on an Ubuntu system. Please refer to the following links for installation instructions:

- [Mininet Installation Guide](https://mininet.org/download/)
- [Ryu Installation Guide](https://ryu.readthedocs.io/en/latest/getting_started.html)

---

## Usage

### Mininet Topology Setup

Navigate to the Mininet network directory and start the custom topology:

```bash
cd mininet_nets/
sudo python custom_topology_v1.py
```

This command will:

- Start Mininet
- Initialize the network topology
- Launch the Mininet CLI

---

### Ryu Controller Setup

Navigate to the controller directory:

```bash
cd ryu_controller/
```

#### Install Requirements

Install the system-wide dependencies listed in `ryu_requirements.txt`.

#### Start Controllers

```bash
./start_controllers.sh
```

#### Stop Controllers

```bash
./stop_controllers.sh
```

---

### Reinforcement Learning (RL) Model Setup

#### Create Python Virtual Environment

```bash
python3.7 -m venv venvrl
```

#### Activate the Environment

```bash
source venvrl/bin/activate
```

#### Install Python Dependencies

```bash
pip install -r rl_requirements.txt
```

#### Train the RL Model

To execute the training script:

```bash
cd model/
python train_decentralized.py
```

The neural network weights are stored in `saved_models/`.

#### Test the RL Model

After specifying the correct weights path, you can start the testing phase:

```bash
cd model/
python test_decentralized.py
```

### Traffic trace execution

Download the 5GB TCP trace available at Netresec:
https://digitalcorpora.org/corpora/network-packet-dumps/

Update the MAC addresses so that they match the hosts in the Mininet network using the following command:

```bash
tcprewrite --enet-dmac=<dst_MAC> --enet-smac=<src_MAC> --infile=<input_pcap> --outfile=<output_pcap>
```

The generated traffic traces for each sending host can be placed in the `mininet_nets/` directory and can be replayed via:

```bash
sudo tcpreplay -i h1-eth0 --stats=1 <h1_filename>.pcap
```

If scripts are preferred to run the traffic, update the `start_traffic.py` script accordingly to reference the traffic scripts.

Once the network is up, you can start the traffic inside the Mininet CLI in the following way:

```bash
py exec(open('start_traffic.py').read())
```
