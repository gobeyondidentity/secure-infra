# UROM Program Samples

DOCA UROM program samples can run only on the host side and require at least one DOCA UROM service instance to be running on BlueField.

The environment variable `DOCA_UROM_SERVICE_FILE` should be set to the path of the UROM service file.

---

# UROM Ping Pong

This sample illustrates how to properly initialize the DOCA UROM interfaces and use its API to create two different workers and run ping pong between them using the Sandbox plugin-based UCX.

The sample uses Open MPI to launch two different processes: one process as a server and the second as a client. The flow is determined by the process rank.

---

## Sample Logic Per Process

1. **Initializing MPI**
2. **Opening the DOCA IB Device**
3. **Creating and Starting UROM Service Context**
4. **Initiating the Sandbox Plugin Host Interface** by attaching the generated plugin ID.
5. **Creating and Starting UROM Worker Context**
6. **Creating and Starting Domain Context**
   - Through the domain context, the sample processes exchange the worker's details to communicate on the BlueField side for the ping pong flow.
7. **Starting Ping Pong Flow**
   - Each process offloads commands to its worker on the BlueField side.
8. **Verifying Ping Pong Completion**
9. **Destroying Contexts**:
   - Domain context
   - Worker context
   - Service context

---

## References

- `urom_ping_pong_sample.c`
- `urom_ping_pong_main.c`
- `meson.build`