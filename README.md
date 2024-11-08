# nas-nsnet2

## Description

This project intends to determine the optimal mixed precision quantization configuration of an NSNet2, a denoising neural network.

The optimal configuration is obtained through the execution of a genetic algorithm, which detects the Pareto optimal solutions considering three metrics:

- PESQ (speech quality metric in the audio file)
- inference time
- memory footprint

Mixed precision occurs fine-grained on each network operation, considering int8, int16, and int32 as possible data types.

## Structure

The repository is divided into the following sections:

- examples: contains examples of the individual parts of the experiment (e.g. the evaluation of the PESQ score).
- nas: contains the script related to the genetic algorithm in Pymoo.
- nsnet2: contains the NSNet2 pytorch model, both the baseline in float32 and the quantized model with mixed precision.
- nsnet2_c: contains the C model of NSNet2 quantized with mixed precision.
- objective function: contains the code relating to the objective function of the genetic algorithm and the evaluation of its metrics (PESQ, inference time, memory footprint).
- plot: contains the script for plotting the results.

## Setup

To setup the repository, first go to the repository top folder:

```bash
cd /path/to/repository/
```

Create and activate the environment:

```bash
python3 -m venv nas_nsnet2_env
source nas_nsnet2_env/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

## Usage

Run the genetic algorithm using:

```bash
python run_nas.py
```

Its execution can last several hours, depending on the specifications of the device.

## License

This project is developed according to MIT license (see LICENSE).