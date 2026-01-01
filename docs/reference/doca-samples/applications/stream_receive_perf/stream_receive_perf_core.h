/*
 * Copyright (c) 2025 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef STREAM_RECEIVE_PERF_CORE_H
#define STREAM_RECEIVE_PERF_CORE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdnoreturn.h>
#include <unistd.h>

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_dev.h>
#include <doca_mmap.h>
#include <doca_pe.h>
#include <doca_rmax.h>

#define APP_NAME "doca_stream_receive_perf"
#define MAX_BUFFERS 2 /* Maximum number of buffers allowed */

/* Scatter type enum for packet processing */
enum scatter_type {
	SCATTER_TYPE_RAW, /* Access to full raw packets, including the network headers */
	SCATTER_TYPE_ULP  /* Access to the contents of the L4 network layer */
};

/* Timestamp format for packet processing */
enum timestamp_format {
	TIMESTAMP_FORMAT_RAW_COUNTER,  /* Raw counter for timestamps */
	TIMESTAMP_FORMAT_FREE_RUNNING, /* Free-running timestamp */
	TIMESTAMP_FORMAT_PTP_SYNCED    /* PTP (Precision Time Protocol)-synced timestamp */
};

/* Application configuration used to customize the application behavior */
struct app_config {
	bool list;				      /* Whether to list available devices */
	bool dump;				      /* Whether to dump packet content */
	enum scatter_type scatter_type;		      /* Scatter type for processing packets */
	enum timestamp_format tstamp_format;	      /* Timestamp format to apply */
	struct in_addr src_ip;			      /* Source IP address to read from */
	struct in_addr dst_ip;			      /* Destination IP address to bind to */
	struct in_addr dev_ip;			      /* IP address of the device to use */
	uint16_t dst_port;			      /* Destination port to bind to */
	uint16_t data_size;			      /* Data (payload) size of the packet */
	uint16_t hdr_size;			      /* Header size of the packet */
	uint32_t num_elements;			      /* Number of elements in the stream buffer */
	bool affinity_mask_set;			      /* Whether a CPU affinity mask is set */
	struct doca_rmax_cpu_affinity *affinity_mask; /* CPU affinity mask */
	useconds_t sleep_us;  /* Sleep duration between packet processing steps (in microseconds) */
	uint32_t min_packets; /* Minimum number of packets to process in a single step */
	uint32_t max_packets; /* Maximum number of packets to process in a single step */
};

/* Global resources required by the application */
struct globals {
	struct doca_mmap *mmap;		      /* Memory map for application resources */
	struct doca_buf_inventory *inventory; /* Buffer inventory for managing memory buffers */
	struct doca_pe *pe;		      /* Progress engine for processing tasks */
};

/* Stream data structure for managing the state and data associated with streaming */
struct stream_data {
	size_t num_buffers;		    /* Number of buffers used*/
	struct timespec start;		    /* Start time of the streaming process */
	struct doca_rmax_in_stream *stream; /* Stream object for receiving data */
	struct doca_buf *buffer;	    /* Memory buffer linked to the stream */
	struct doca_rmax_flow *flow;	    /* Flow object attached to the stream */
	uint16_t pkt_size[MAX_BUFFERS];	    /* Size of an element in each buffers */
	uint16_t stride_size[MAX_BUFFERS];  /* Stride size for packet data in the buffers */
	/* statistics */
	size_t recv_pkts;  /* Number of packets received */
	size_t recv_bytes; /* Total number of bytes received */
	/* control flow */
	bool dump;	    /* Whether to dump the content of received packets */
	bool run_recv_loop; /* Flag to indicate whether the receive loop should continue running */
};

/*
 * Initializes the application configuration with default values.
 * Also creates the CPU affinity mask used for assigning tasks to specific cores
 *
 * @config [in/out]: Pointer to the application configuration structure to initialize
 * @return: true on successful initialization; false otherwise
 */
bool init_config(struct app_config *config);

/*
 * Releases resources allocated by the application configuration.
 * This includes destroying the CPU affinity mask to clean up system resources
 *
 * @config [in]: Pointer to the application configuration structure to destroy
 */
void destroy_config(struct app_config *config);

/*
 * Registers all configurable parameters for handling command-line arguments.
 * These parameters help customize the application's behavior
 *
 * @return: true on successful registration of all parameters; false otherwise
 */
bool register_argp_params(void);

/*
 * Verifies that all mandatory arguments (e.g., IP addresses and port) are set
 * in the application configuration before proceeding
 *
 * @config [in]: Pointer to the application configuration structure to validate
 * @return: true if all mandatory arguments are set; false otherwise
 */
bool mandatory_args_set(struct app_config *config);

/*
 * Enumerates and lists all available devices that the application
 * can interact with, along with their details like network interface name, PCI address,
 * and IP address
 */
void list_devices(void);

/*
 * Opens a specific device based on its IP address. Uses the DOCA library
 * to locate the device, open it, and return a device handle
 *
 * @dev_ip [in]: Pointer to the IP address of the device to open
 * @return: Pointer to the opened device on success; NULL otherwise
 */
struct doca_dev *open_device(struct in_addr *dev_ip);

/*
 * Initializes global application resources like memory maps, progress engines,
 * and buffer inventories. These resources are essential for handling stream operations
 *
 * @config [in]: Pointer to the application configuration structure
 * @dev [in]: Pointer to the device handle opened by the application
 * @globals [in/out]: Pointer to the global resources structure to initialize
 * @return: DOCA_SUCCESS on successful initialization; appropriate DOCA error otherwise
 */
doca_error_t init_globals(struct app_config *config, struct doca_dev *dev, struct globals *globals);

/*
 * Releases and destroys global application resources initialized earlier. This
 * includes stopping memory maps, destroying buffer inventories, and releasing progress engines
 *
 * @globals [in]: Pointer to the global resources structure to destroy
 * @dev [in]: Pointer to the device handle associated with these resources
 * @return: true if resources were successfully destroyed; false otherwise
 */
bool destroy_globals(struct globals *globals, struct doca_dev *dev);

/*
 * Configures and initializes a stream for receiving packets.
 * Sets stream parameters, links to memory buffers, and connects to a flow for data reception
 *
 * @config [in]: Pointer to the application configuration structure
 * @dev [in]: Pointer to the device handle opened by the application
 * @globals [in]: Pointer to the global resources structure
 * @data [in/out]: Pointer to the stream data structure to initialize
 * @return: DOCA_SUCCESS on successful initialization; appropriate DOCA error otherwise
 */
doca_error_t init_stream(struct app_config *config,
			 struct doca_dev *dev,
			 struct globals *globals,
			 struct stream_data *data);

/*
 * Cleans up and destroys a stream, including flow detachment, stopping the context,
 * buffer release, and stream destruction
 *
 * @dev [in]: Pointer to the device handle associated with the stream
 * @globals [in]: Pointer to the global resources structure
 * @data [in/out]: Pointer to the stream data structure to destroy
 * @return: true if all resources are successfully destroyed; false otherwise
 */
bool destroy_stream(struct doca_dev *dev, struct globals *globals, struct stream_data *data);

/*
 * Runs the packet reception loop. Progresses the DOCA progress engine,
 * collects statistics periodically, and handles sleep intervals if specified.
 * The loop runs until the `run_recv_loop` flag in the stream_data structure is set to false
 *
 * @config [in]: Pointer to the application configuration structure, defining loop behavior such as sleep intervals
 * @globals [in]: Pointer to the global resources structure, including progress engine
 * @data [in/out]: Pointer to the stream data structure, managing control flow and maintaining statistics
 * @return: true if the loop runs and exits successfully; false otherwise
 */
bool run_recv_loop(const struct app_config *config, struct globals *globals, struct stream_data *data);

#endif // STREAM_RECEIVE_PERF_CORE_H
