## Graph Plugin

This plugin provides a simple example for creating a UROM plugin interface. It exposes only a single command, `loopback`, which:
- Sends a specific value in the command.
- Expects to receive the same value in the notification from the UROM worker.

## References

- `/meson.build`
- `urom_graph.h`
- `worker_graph.c`
- `worker_graph.h`

