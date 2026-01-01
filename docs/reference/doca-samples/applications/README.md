# [DOCA Reference Applications](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-Applications)

This page provides an overview of the DOCA reference applications implemented on top of NVIDIA® BlueField®.

All of the DOCA reference applications described in this section are governed under the BSD-3 software license agreement.

## [App Shield Agent](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-AppShieldAgent)
The [DOCA App Shield Agent](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+DPA+All-to-all+Application+Guide) reference application describes how to build secure process monitoring and is based on the DOCA APSH library, which leverages DPU capabilities such as regular expression (RXP) acceleration engine, hardware-based DMA, and more.

## [DMA Copy](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-DMACopy)
The [DOCA DMA Copy](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+DMA+Copy+Application+Guide) reference application describes how to transfer files between the DPU and the host. The application is based on the direct memory access (DMA) library, which leverages hardware acceleration for data copy for both local and remote memory.

## [DPA All-to-all](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-DPAAll-to-all)
The [DOCA DPA All-to-all](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+DPA+All-to-all+Application+Guide) reference application is a collective operation that allows data to be copied between multiple processes. This application is implemented using DOCA DPA, which leverages the data path accelerator (DPA) inside the BlueField-3 to offload the copying of the data to the DPA and leave the CPU free for other computations.

## [East-West Overlay Encryption](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-East-WestOverlayEncryption)
The [DOCA East-West Overlay Encryption](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+East-West+Overlay+Encryption+Application) reference application (IPsec) sets up encrypted connections between different devices and works by encrypting IP packets and authenticating the packets' originator. It is based on a strongSwan solution, which is an open-source IPsec-based VPN solution.

## [Ethernet L2 Forwarding](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-EthernetL2Forwarding)
The [DOCA Ethernet L2 Forwarding](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+Eth+L2+Forwarding+Application+Guide) reference application forwards traffic from a single RX port to a single TX port and vice versa, leveraging DOCA's task/event batching feature for enhanced performance.

## [File Compression](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-FileCompression)
The [DOCA File Compression](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+File+Compression+Application+Guide) reference application shows how to compress and decompress data using hardware acceleration and to send and receive it. The application is based on the DOCA Compress and DOCA Comm-Channel libraries.

## [File Integrity](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-FileIntegrity)
The [DOCA File Integrity](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+File+Integrity+Application+Guide) reference application shows how to send and receive files in a secure way using the hardware Crypto engine. It is based on the DOCA SHA and DOCA Comm-Channel libraries.

## [GPU Packet Processing](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-GPUPacketProcessing)
The [DOCA GPU Packet Processing](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+GPU+Packet+Processing+Application+Guide) reference application shows how to combine DOCA GPUNetIO, DOCA Ethernet, and DOCA Flow to manage ICMP, UDP, TCP, and HTTP connections with a GPU-centric approach using CUDA kernels without involving the CPU in the main data path.

## [IPsec Gateway](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-IPsecGateway)
The [DOCA IPsec Gateway](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+IPsec+Security+Gateway+Application+Guide) reference application demonstrates how to insert rules related to IPsec encryption and decryption based on the DOCA Flow and IPsec libraries, which leverage the DPU's hardware capability for secure network communication.

## [NVMe Emulation](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-NVMeEmulation)
The [DOCA NVMe Emulation](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+NVMe+Emulation+App+Guide) reference application exhibits how to use the [DOCA DevEmu PCI Generic API](https://docs.nvidia.com/doca/archive/2-9-0/DOCA+DevEmu+PCI+Generic) along with SPDK to emulate an NVMe PCIe function using hardware acceleration to fully emulate the storage device.

## [Programmable Congestion Control](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-ProgrammableCongestionControl)
The [DOCA Programmable Congestion Control](httphttps://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+PCC+Application+Guide) reference application is based on the DOCA PCC library and allows users to design and implement their own congestion control algorithm, giving them good flexibility to work out an optimal solution to handle congestion in their clusters.

## [PSP Gateway](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-PSPGateway)
The [DOCA PSP Gateway](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+PSP+Gateway+Application+Guide) reference application demonstrates how to exchange keys between application instances and insert rules controlling PSP encryption and decryption using the DOCA Flow library.

## [Secure Channel](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-SecureChannel)
The [DOCA Secure Channel](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+Secure+Channel+Application+Guide) reference application is used to establish a secure, network-independent communication channel between the host and the DPU based on the DOCA Comm Channel library.

## [Simple Forward VNF](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-SimpleForwardVNF)
The [DOCA Simple Forward VNF](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+Simple+Forward+VNF+Application+Guide) reference application is a forwarding application that takes VXLAN traffic from a single RX port and transmits it on a single TX port. It is based on the DOCA Flow library, which leverages DPU capabilities such as building generic execution pipes in the hardware, and more.

## [Storage Zero Copy](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-StorageZeroCopy)
The [DOCA Storage Zero Copy](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+Storage+Zero+Copy) reference applications demonstrate a way to leverage hardware acceleration to implement a simple data storage solution that allows for data to be stored and retrieved efficiently without any unnecessary copying of data.

## [Switch](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-Switch)
The [DOCA Switch](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+Switch+Application+Guide) reference application is used to establish internal switching between representor ports on the DPU. It is based on the DOCA Flow library, which leverages DPU capabilities such as building generic execution pipes in the hardware, and more.

## [UROM RDMO](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-UROMRDMO)
The [DOCA UROM RDMO](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+UROM+RDMO+Application+Guide) reference application demonstrates how to execute an Active Message outside the context of the target process. It is based on the DOCA UROM (Unified Resources and Offload Manager) library as a framework to launch UROM workers on the DPU and uses the UCX communication framework, which leverages the DPU's low-latency and high-bandwidth utilization of its network engine.

## [YARA Inspection](https://docs.nvidia.com/doca/archive/2-9-0/doca+reference+applications/index.html#src-3095356375_id-.DOCAReferenceApplicationsv2.9.0LTS-YARAInspection)
The [DOCA YARA Inspection](https://docs.nvidia.com/doca/archive/2-9-0/NVIDIA+DOCA+YARA+Inspection+Application+Guide) reference application describes how to build YARA rule inspection for processes and is based on the DOCA APSH library, which leverages DPU capabilities such as the regular expression (RXP) acceleration engine, hardware-based DMA, and more.
