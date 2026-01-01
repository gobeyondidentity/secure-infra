/*
 * Copyright (c) 2022 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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

#include <string.h>
#include <unistd.h>

#include <doca_log.h>
#include <doca_flow.h>
#include <doca_bitfield.h>

#include "flow_common.h"
#include <rte_byteorder.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_net.h>
#include "flow_mark_set.h"

DOCA_LOG_REGISTER(FLOW_MARK_SET);

#define MAX_PKTS 16
#define WAIT_SECS 15

static doca_error_t create_match_pipe(struct doca_flow_port *port,
				      struct doca_flow_pipe **pipe,
				      struct doca_flow_pipe *fwd_pipe,
				      struct mark_set_config *app_cfg)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions, *actions_arr[NB_ACTIONS_ARR];
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg *pipe_cfg;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));

	match.parser_meta.outer_l3_type = DOCA_HTOBE32(0xffffffff);
	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.ip4.src_ip = BE_IPV4_ADDR(255, 255, 255, 255);
	if (app_cfg->action_changeable)
		actions.meta.mark = UINT32_MAX;
	else
		actions.meta.mark = DOCA_HTOBE32(app_cfg->mark_value);

	actions_arr[0] = &actions;

	result = doca_flow_pipe_cfg_create(&pipe_cfg, port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		return result;
	}

	result = set_flow_pipe_cfg(pipe_cfg, "MATCH_PIPE", DOCA_FLOW_PIPE_BASIC, true);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg match: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	result = doca_flow_pipe_cfg_set_actions(pipe_cfg, actions_arr, NULL, NULL, NB_ACTIONS_ARR);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg monitor: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	/* forwarding traffic to other port */
	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = fwd_pipe;

	result = doca_flow_pipe_create(pipe_cfg, &fwd, NULL, pipe);
destroy_pipe_cfg:
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	return result;
}

static doca_error_t add_match_pipe_entry(struct doca_flow_pipe *pipe,
					 struct entries_status *status,
					 struct mark_set_config *app_cfg)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_pipe_entry *entry;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));

	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.ip4.src_ip = BE_IPV4_ADDR(1, 2, 3, 4);
	if (app_cfg->action_changeable) {
		actions.meta.mark = DOCA_HTOBE32(app_cfg->mark_value);
	}
	actions.action_idx = 0;

	result = doca_flow_pipe_add_entry(0, pipe, &match, &actions, NULL, NULL, 0, status, &entry);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

static doca_error_t create_rss_pipe(struct doca_flow_port *port, struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_monitor monitor;
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg *pipe_cfg;
	uint16_t rss_queues[1];
	doca_error_t result;
	struct doca_flow_actions actions, *actions_arr[NB_ACTIONS_ARR];

	memset(&match, 0, sizeof(match));
	memset(&fwd, 0, sizeof(fwd));
	memset(&monitor, 0, sizeof(monitor));
	memset(&actions, 0, sizeof(actions));

	result = doca_flow_pipe_cfg_create(&pipe_cfg, port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		return result;
	}

	result = set_flow_pipe_cfg(pipe_cfg, "RSS_PIPE", DOCA_FLOW_PIPE_BASIC, false);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg match: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	actions_arr[0] = &actions;
	result = doca_flow_pipe_cfg_set_actions(pipe_cfg, actions_arr, NULL, NULL, NB_ACTIONS_ARR);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg monitor: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &monitor);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg monitor: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	/* RSS queue - send matched traffic to queue 0  */
	rss_queues[0] = 0;
	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
	fwd.rss.queues_array = rss_queues;
	fwd.rss.inner_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_TCP;
	fwd.rss.nr_queues = 1;

	result = doca_flow_pipe_create(pipe_cfg, &fwd, NULL, pipe);
destroy_pipe_cfg:
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	return result;
}

static doca_error_t add_rss_pipe_entry(struct doca_flow_pipe *pipe, struct entries_status *status)
{
	struct doca_flow_pipe_entry *entry;
	struct doca_flow_actions actions;
	struct doca_flow_match match;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));

	actions.action_idx = 0;

	result = doca_flow_pipe_add_entry(0, pipe, &match, &actions, NULL, NULL, 0, status, &entry);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

static void handle_rx_tx_pkts(uint32_t port_id, uint16_t nb_queues)
{
	int rc;
	uint32_t queue_id;
	uint32_t secs = WAIT_SECS;
	uint32_t nb_rx;
	uint32_t i;
	uint32_t sw_packet_type;
	uint32_t nb_tx;
	uint32_t retry;
	struct rte_mbuf *mbufs[MAX_PKTS];

	rc = rte_flow_dynf_metadata_register();
	if (unlikely(rc)) {
		DOCA_LOG_ERR("Enable metadata failed");
		return;
	}

	while (secs--) {
		sleep(1);
		for (queue_id = 0; queue_id < nb_queues; queue_id++) {
			nb_rx = rte_eth_rx_burst(port_id, queue_id, mbufs, MAX_PKTS);
			for (i = 0; i < nb_rx; i++) {
				sw_packet_type = rte_net_get_ptype(mbufs[i], NULL, RTE_PTYPE_ALL_MASK);
				if (mbufs[i]->ol_flags & RTE_MBUF_F_RX_FDIR_ID) {
					DOCA_LOG_INFO("packet mark value: 0x%x", mbufs[i]->hash.fdir.hi);
					mbufs[i]->ol_flags |= RTE_MBUF_DYNFLAG_TX_METADATA;
				} else {
					DOCA_LOG_INFO("Pkt received: 0x%x\n", sw_packet_type);
				}

				retry = 0;
				do {
					nb_tx = rte_eth_tx_burst(port_id, queue_id, &mbufs[i], 1);
					if (nb_tx == 1)
						break;
					else
						rte_delay_us(50000);
				} while (++retry < 5);
				rte_pktmbuf_free(mbufs[i]);
			}
		}
	}
}

doca_error_t flow_mark_set(int nb_queues, struct mark_set_config *app_cfg)
{
	int nb_ports = 2;
	struct flow_resources resource = {0};
	uint32_t nr_shared_resources[SHARED_RESOURCE_NUM_VALUES] = {0};
	struct doca_flow_port *ports[nb_ports];
	struct doca_dev *dev_arr[nb_ports];
	int num_of_entries = 2;
	uint32_t actions_mem_size[nb_ports];
	doca_error_t result;
	int port_id;
	struct entries_status status;
	struct doca_flow_pipe *rss_pipe;
	struct doca_flow_pipe *match_pipe;

	resource.nr_counters = 16;
	result = init_doca_flow(nb_queues, "vnf,hws", &resource, nr_shared_resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA Flow: %s", doca_error_get_descr(result));
		return result;
	}

	memset(dev_arr, 0, sizeof(struct doca_dev *) * nb_ports);
	ARRAY_INIT(actions_mem_size, ACTIONS_MEM_SIZE(nb_queues, num_of_entries));
	result = init_doca_flow_ports(nb_ports, ports, true, dev_arr, actions_mem_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA ports: %s", doca_error_get_descr(result));
		doca_flow_destroy();
		return result;
	}

	for (port_id = 0; port_id < nb_ports; port_id++) {
		memset(&status, 0, sizeof(status));

		result = create_rss_pipe(ports[port_id], &rss_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create rss pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_rss_pipe_entry(rss_pipe, &status);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add entry: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = create_match_pipe(ports[port_id], &match_pipe, rss_pipe, app_cfg);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create classifier pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_match_pipe_entry(match_pipe, &status, app_cfg);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add entry to classifier pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = doca_flow_entries_process(ports[port_id], 0, DEFAULT_TIMEOUT_US, num_of_entries);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to process entries: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		if (status.nb_processed != num_of_entries || status.failure) {
			DOCA_LOG_ERR("Failed to process entries: %d, %d, %d",
				     status.nb_processed,
				     num_of_entries,
				     status.failure);
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return DOCA_ERROR_BAD_STATE;
		}
	}

	DOCA_LOG_INFO("Wait few seconds for packets to arrive");
	handle_rx_tx_pkts(0, nb_queues);
	DOCA_LOG_INFO("===================================================");

	result = stop_doca_flow_ports(nb_ports, ports);
	doca_flow_destroy();
	return result;
}
