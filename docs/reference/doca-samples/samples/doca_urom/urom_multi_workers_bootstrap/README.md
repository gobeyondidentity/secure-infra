# UROM Program Samples

DOCA UROM program samples can run only on the host side and require at least one DOCA UROM service instance to be running on BlueField.

The environment variable `DOCA_UROM_SERVICE_FILE` should be set to the path of the UROM service file.

---

## UROM Multi-worker Bootstrap

This sample illustrates how to properly initialize DOCA UROM interfaces and use the API to spawn multiple workers on the same application process.

### Overview

The sample initiates four threads as UROM workers to execute concurrently, alongside the main thread operating as a UROM service. It divides the workers into two groups based on their IDs, with odd-numbered workers in one group and even-numbered workers in the other.

Each worker executes the data loopback command by using the Graph plugin, sends a specific value, and expects to receive the same value in the notification.

### The sample logic includes

1. **Opening the DOCA IB Device**
2. **Initializing Necessary DOCA Core Structures**
3. **Creating and Starting UROM Service Context**
4. **Initiating the Graph Plugin Host Interface** by attaching the generated plugin ID.
5. **Launching 4 Threads**:
   - For each thread:
     - Create and start the UROM worker context.
     - Once the worker context switches to running, send the loopback graph command and wait for a notification.
     - Verify the received data.
6. **Waiting for Interrupt Signal**
7. **Main Thread Responsibilities**:
   - Check for pending jobs of spawning workers (4 jobs, one per thread).
   - Check for pending jobs of destroying workers (4 jobs, one per thread) for exiting.
8. **Cleaning Up and Exiting**

---

## References

- `urom_multi_workers_bootstrap_sample.c`
- `urom_multi_workers_bootstrap_main.c`
- `meson.build`

