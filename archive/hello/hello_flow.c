/*
 * DOCA Flow Hello World
 *
 * Demonstrates basic DOCA Flow initialization and pipe creation concepts.
 * This is a conceptual example showing the API flow without DPDK packet processing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_flow.h>

DOCA_LOG_REGISTER(FLOW_HELLO);

#define MAX_PORT_STR_LEN 128

/*
 * Initialize DOCA Flow library
 */
static doca_error_t init_doca_flow(uint16_t nb_queues)
{
    struct doca_flow_cfg flow_cfg = {0};
    doca_error_t result;

    /* Configure DOCA Flow */
    flow_cfg.pipe_queues = nb_queues;
    flow_cfg.mode_args = "vnf,hws"; /* VNF mode with hardware steering */
    flow_cfg.resource.nb_counters = 1024;

    DOCA_LOG_INFO("Initializing DOCA Flow with %u queues", nb_queues);

    result = doca_flow_init(&flow_cfg);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to initialize DOCA Flow: %s", doca_error_get_descr(result));
        return result;
    }

    DOCA_LOG_INFO("DOCA Flow initialized successfully");
    return DOCA_SUCCESS;
}

/*
 * Create a DOCA Flow port
 */
static doca_error_t create_flow_port(struct doca_dev *dev,
                                      uint16_t port_id,
                                      struct doca_flow_port **port)
{
    struct doca_flow_port_cfg port_cfg = {0};
    char port_id_str[MAX_PORT_STR_LEN];
    doca_error_t result;

    snprintf(port_id_str, MAX_PORT_STR_LEN, "%u", port_id);
    port_cfg.port_id = port_id;
    port_cfg.type = DOCA_FLOW_PORT_DPDK_BY_ID;
    port_cfg.devargs = port_id_str;
    port_cfg.dev = dev;

    DOCA_LOG_INFO("Creating DOCA Flow port %u", port_id);

    result = doca_flow_port_start(&port_cfg, port);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start port: %s", doca_error_get_descr(result));
        return result;
    }

    DOCA_LOG_INFO("Port %u created successfully", port_id);
    return DOCA_SUCCESS;
}

/*
 * Create a simple forwarding pipe
 *
 * This pipe matches on IPv4 destination address and forwards to a port
 */
static doca_error_t create_simple_fwd_pipe(struct doca_flow_port *port,
                                            struct doca_flow_pipe **pipe)
{
    struct doca_flow_match match = {0};
    struct doca_flow_actions actions = {0};
    struct doca_flow_actions *actions_arr[1];
    struct doca_flow_fwd fwd = {0};
    struct doca_flow_pipe_cfg *pipe_cfg;
    doca_error_t result;

    DOCA_LOG_INFO("Creating simple IPv4 forwarding pipe");

    /* Match on IPv4 destination address (wildcard - will be set in entries) */
    match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    match.outer.ip4.dst_ip = 0xffffffff; /* Match mask - will match any IP */

    actions_arr[0] = &actions;

    /* Create pipe configuration */
    result = doca_flow_pipe_cfg_create(&pipe_cfg, port);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create pipe cfg: %s", doca_error_get_descr(result));
        return result;
    }

    /* Set pipe name and type */
    result = doca_flow_pipe_cfg_set_name(pipe_cfg, "SIMPLE_FWD_PIPE");
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set pipe name: %s", doca_error_get_descr(result));
        goto destroy_cfg;
    }

    result = doca_flow_pipe_cfg_set_type(pipe_cfg, DOCA_FLOW_PIPE_BASIC);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set pipe type: %s", doca_error_get_descr(result));
        goto destroy_cfg;
    }

    result = doca_flow_pipe_cfg_set_is_root(pipe_cfg, true);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set pipe as root: %s", doca_error_get_descr(result));
        goto destroy_cfg;
    }

    /* Set match criteria */
    result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set match: %s", doca_error_get_descr(result));
        goto destroy_cfg;
    }

    /* Set actions */
    result = doca_flow_pipe_cfg_set_actions(pipe_cfg, actions_arr, NULL, NULL, 1);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set actions: %s", doca_error_get_descr(result));
        goto destroy_cfg;
    }

    /* Forward to port */
    fwd.type = DOCA_FLOW_FWD_PORT;
    fwd.port_id = 0;

    /* Create the pipe */
    result = doca_flow_pipe_create(pipe_cfg, &fwd, NULL, pipe);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create pipe: %s", doca_error_get_descr(result));
        goto destroy_cfg;
    }

    DOCA_LOG_INFO("Simple forwarding pipe created successfully");

destroy_cfg:
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
}

/*
 * Add a flow entry to the pipe
 */
static doca_error_t add_flow_entry(struct doca_flow_pipe *pipe,
                                    uint32_t dst_ip,
                                    struct doca_flow_pipe_entry **entry)
{
    struct doca_flow_match match = {0};
    struct doca_flow_actions actions = {0};
    struct doca_flow_pipe_entry *tmp_entry;
    doca_error_t result;
    char ip_str[32];

    /* Convert IP to string for logging */
    snprintf(ip_str, sizeof(ip_str), "%u.%u.%u.%u",
             (dst_ip >> 24) & 0xFF,
             (dst_ip >> 16) & 0xFF,
             (dst_ip >> 8) & 0xFF,
             dst_ip & 0xFF);

    DOCA_LOG_INFO("Adding flow entry for destination IP: %s", ip_str);

    /* Set specific IP to match */
    match.outer.ip4.dst_ip = dst_ip;

    /* Add entry */
    result = doca_flow_pipe_add_entry(0, pipe, &match, &actions, NULL, NULL, 0, NULL, &tmp_entry);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to add entry: %s", doca_error_get_descr(result));
        return result;
    }

    /* Process the entry */
    result = doca_flow_entries_process(doca_flow_pipe_to_port(pipe), 0, 1000, 0);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to process entries: %s", doca_error_get_descr(result));
        return result;
    }

    *entry = tmp_entry;
    DOCA_LOG_INFO("Flow entry added successfully");
    return DOCA_SUCCESS;
}

/*
 * Main hello world function
 */
int main(void)
{
    struct doca_devinfo **dev_list = NULL;
    struct doca_dev *dev = NULL;
    struct doca_flow_port *port = NULL;
    struct doca_flow_pipe *pipe = NULL;
    struct doca_flow_pipe_entry *entry = NULL;
    uint32_t nb_devs = 0;
    doca_error_t result;

    printf("=================================\n");
    printf("   DOCA Flow Hello World        \n");
    printf("=================================\n\n");

    /* Find DOCA devices */
    result = doca_devinfo_create_list(&dev_list, &nb_devs);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create device list: %s", doca_error_get_descr(result));
        return EXIT_FAILURE;
    }

    if (nb_devs == 0) {
        DOCA_LOG_INFO("No DOCA devices found");
        DOCA_LOG_INFO("Note: DOCA Flow requires actual network hardware (ConnectX/Bluefield)");
        goto cleanup;
    }

    DOCA_LOG_INFO("Found %u DOCA device(s)", nb_devs);

    /* Open first device */
    result = doca_dev_open(dev_list[0], &dev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to open device: %s", doca_error_get_descr(result));
        goto cleanup;
    }

    DOCA_LOG_INFO("Device opened successfully");

    /* Initialize DOCA Flow */
    result = init_doca_flow(1);
    if (result != DOCA_SUCCESS) {
        goto cleanup_dev;
    }

    /* Create port */
    result = create_flow_port(dev, 0, &port);
    if (result != DOCA_SUCCESS) {
        goto cleanup_flow;
    }

    /* Create pipe */
    result = create_simple_fwd_pipe(port, &pipe);
    if (result != DOCA_SUCCESS) {
        goto cleanup_port;
    }

    /* Add example flow entry (match packets to 192.168.1.1) */
    result = add_flow_entry(pipe, 0xC0A80101, &entry); /* 192.168.1.1 */
    if (result != DOCA_SUCCESS) {
        goto cleanup_port;
    }

    printf("\n=================================\n");
    printf("DOCA Flow Hello World completed!\n");
    printf("Created:\n");
    printf("  - 1 Flow port\n");
    printf("  - 1 IPv4 forwarding pipe\n");
    printf("  - 1 Flow entry (dst IP: 192.168.1.1)\n");
    printf("=================================\n");

    /* Cleanup */
cleanup_port:
    if (port != NULL)
        doca_flow_port_stop(port);

cleanup_flow:
    doca_flow_destroy();

cleanup_dev:
    if (dev != NULL)
        doca_dev_close(dev);

cleanup:
    if (dev_list != NULL)
        doca_devinfo_destroy_list(dev_list);

    return (result == DOCA_SUCCESS) ? EXIT_SUCCESS : EXIT_FAILURE;
}
