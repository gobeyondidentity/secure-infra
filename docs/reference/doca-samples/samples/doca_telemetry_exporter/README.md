# DOCA Telemetry Exporter Samples

These samples demonstrate the usage of the DOCA Telemetry Exporter API, including custom telemetry schema and NetFlow functionalities.

## Telemetry Export

This sample illustrates how to use the telemetry exporter API with a custom schema.

### Sample Logic:
1. Configuring schema attributes.
2. Initializing the schema.
3. Creating a telemetry exporter source.
4. Creating example events.
5. Reporting the example events via DOCA Telemetry Exporter.
6. Destroying the source and schema.

### References:
- `telemetry_export/telemetry_export_sample.c`
- `telemetry_export/telemetry_export_main.c`
- `telemetry_export/meson.build`

---

## Telemetry Export NetFlow

This sample demonstrates how to use the NetFlow functionality of the telemetry exporter API.

### Sample Logic:
1. Configuring NetFlow attributes.
2. Initializing NetFlow.
3. Creating a telemetry exporter source.
4. Starting NetFlow.
5. Creating example events.
6. Reporting the example events via DOCA Telemetry Exporter.
7. Destroying NetFlow.

### References:
- `telemetry_export_netflow/telemetry_export_netflow_sample.c`
- `telemetry_export_netflow/telemetry_export_netflow_main.c`
- `telemetry_export_netflowt/meson.build`
