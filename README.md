# Quantized_Neural_Network_Using_Brevitas_on_RISC-V
Quantization-aware training is a more efficient approach for edge AI/ML. Here, the LeNet model is trained using QAT on the MNIST dataset and deployed on a RISC-V machine.
# Host system 
Ubuntu 24.04.2 LTS
# Board details
The quantized LeNet classifier is deployed on a RISC-V 32-bit machine (VSDSquadron PRO Board). More details are available on https://github.com/ShekharShwetank/Quantized_Neural_Network_on_RISC-V. 
# Virtual environment and packages 
```python
python -m venv qnn_venv
source qnn_venv/bin/activate
pip install torch
pip install brevitas
```
Install necessary dependencies, if required. 
# Training using QAT Brevitas and Export QAT model in QDQ ONNX format
```python
python train_convert.py
```
# Freedom Studio Installation and Project Setup
Follow the steps given here, https://github.com/ShekharShwetank/Quantized_Neural_Network_on_RISC-V.
Freedom Studio supports QEMU; hence, it is possible to run code without the hardware.
Start with the example project as shown in https://github.com/ShekharShwetank/Quantized_Neural_Network_on_RISC-V.
Instead of an actual hardware device, select "qemu-sifive-e31", which acts as a RISC-V 32-bit MCU.

<img width="694" height="726" alt="step1" src="https://github.com/user-attachments/assets/30845ae4-e76c-4ccd-9be5-31ecd41c777b" />

It will automatically set up the rest of the options. 
# QEMU configuration
To configure QEMU, open the <project_dir>/bsp/settings.mk file and replace the content with the following:
```
RISCV_ARCH = rv32imac_zicsr_zifencei
RISCV_ABI = ilp32
RISCV_CMODEL = medlow
RISCV_SERIES = sifive-3-series

TARGET_TAGS = qemu
TARGET_DHRY_ITERS = 20000000
TARGET_CORE_ITERS = 5000
TARGET_FREERTOS_WAIT_MS = 1000
TARGET_INTR_WAIT_CYCLE  = 0
```
Then, build the project.
# Testing with QEMU 
Building the project will generate an ELF file under <project_dir>/src/debug/<project_name>.elf. 
To run the code with QEMU, select the proper tool "qemu-system-riscv32.exe" as shown below.

<img width="786" height="531" alt="step2" src="https://github.com/user-attachments/assets/13a02de2-db2e-4708-885a-3806c6ab5298" />

Then, go to the debug configuration under the "Project" menu and double click on "SiFive GDB QEMU Debugging (LEGACY)"; 
It will create a configuration file. Now, click on debug and "Run" the code; you will see the output in the console. 

<img width="1345" height="710" alt="step3" src="https://github.com/user-attachments/assets/8ce84881-0c5a-4873-aef5-1b407b0a3ff3" />

# Running classifier on RISC-V QEMU
I am still working on integrating the quantised model into QEMU. Soon, I will update the steps here.  
# Sources
https://github.com/ShekharShwetank/Quantized_Neural_Network_on_RISC-V

https://github.com/Xilinx/brevitas

