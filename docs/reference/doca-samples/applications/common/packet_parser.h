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

#ifndef PACKET_PARSER_H_
#define PACKET_PARSER_H_

#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#include <doca_error.h>

enum parser_pkt_type {
	PARSER_PKT_TYPE_TUNNELED,
	PARSER_PKT_TYPE_PLAIN,
	PARSER_PKT_TYPE_UNKNOWN,
	PARSER_PKT_TYPE_NUM = PARSER_PKT_TYPE_UNKNOWN,
};

struct link_parser_ctx {
	size_t len;		   /* Total length of headers parsed */
	uint16_t next_proto;	   /* Next protocol */
	struct rte_ether_hdr *eth; /* Ethernet header */
};

struct network_parser_ctx {
	size_t len;	    /* Total length of headers parsed */
	uint8_t ip_version; /* Version of IP */
	uint8_t next_proto; /* Next protocol */
	bool frag;	    /* Flag indicating fragmented packet */
	union {
		struct rte_ipv4_hdr *ipv4_hdr; /* IPv4 header */
		struct {
			struct rte_ipv6_hdr *hdr;		/* IPv6 header */
			struct rte_ipv6_fragment_ext *frag_ext; /* IPv6 fragmentation extension header */
		} ipv6;
	};
};

struct transport_parser_ctx {
	uint8_t proto; /* Protocol ID */
	size_t len;    /* Total length of headers parsed */
	union {
		struct rte_udp_hdr *udp_hdr; /* UDP header, when proto==UDP */
		struct rte_tcp_hdr *tcp_hdr; /* TCP header, when proto==TCP */
	};
};

struct gtp_parser_ctx {
	size_t len;			       /* Total length of headers parsed */
	struct rte_gtp_hdr *gtp_hdr;	       /* GTP protocol header */
	struct rte_gtp_hdr_ext_word *opt_hdr;  /* GTP option header */
	struct rte_gtp_psc_type0_hdr *ext_hdr; /* GTP extension header */
};

struct conn_parser_ctx {
	size_t len;				   /* Total length of headers parsed */
	struct link_parser_ctx link_ctx;	   /* Link-layer parser context */
	struct network_parser_ctx network_ctx;	   /* Network-layer parser context */
	struct transport_parser_ctx transport_ctx; /* Transport-layer parser context */
};

struct tun_parser_ctx {
	size_t len;				   /* Total length of headers parsed */
	struct link_parser_ctx link_ctx;	   /* Link-layer parser context */
	struct network_parser_ctx network_ctx;	   /* Network-layer parser context */
	struct transport_parser_ctx transport_ctx; /* Transport-layer parser context */
	struct gtp_parser_ctx gtp_ctx;		   /* GTP tunnel parser context */
	struct conn_parser_ctx inner;		   /* Tunnel-encapsulated connection parser context */
};

/*
 * Parse link-layer protocol headers
 *
 * @data [in]: pointer to the start of the data
 * @data_end [in]: pointer to the end of the data
 * @ctx [out]: pointer to the parser context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t link_parse(uint8_t *data, uint8_t *data_end, struct link_parser_ctx *ctx);

/*
 * Parse network-layer protocol headers
 *
 * @data [in]: pointer to the start of the data
 * @data_end [in]: pointer to the end of the data
 * @expected_proto [in]: expected network-layer protocol as indicated by upper layer
 * @ctx [out]: pointer to the parser context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t network_parse(uint8_t *data, uint8_t *data_end, uint16_t expected_proto, struct network_parser_ctx *ctx);

/*
 * Parse transport-layer protocol headers
 *
 * @data [in]: pointer to the start of the data
 * @data_end [in]: pointer to the end of the data
 * @proto [in]: transport-layer protocol as indicated by upper layer
 * @ctx [out]: pointer to the parser context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t transport_parse(uint8_t *data, uint8_t *data_end, uint8_t proto, struct transport_parser_ctx *ctx);

/*
 * Parse GPTU tunnel headers
 *
 * @data [in]: pointer to the start of the data
 * @data_end [in]: pointer to the end of the data
 * @ctx [out]: pointer to the parser context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t gtpu_parse(uint8_t *data, uint8_t *data_end, struct gtp_parser_ctx *ctx);

/*
 * Parse the payload headers that identify the connection 5T
 *
 * @data [in]: pointer to the start of the data
 * @data_end [in]: pointer to the end of the data
 * @ctx [out]: pointer to the parser context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t conn_parse(uint8_t *data, uint8_t *data_end, struct conn_parser_ctx *ctx);

/*
 * Parse a plain non-tunneled packet
 *
 * @data [in]: pointer to the start of the data
 * @data_end [in]: pointer to the end of the data
 * @ctx [out]: pointer to the parser context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t plain_parse(uint8_t *data, uint8_t *data_end, struct conn_parser_ctx *ctx);

/*
 * Parse a tunneled packet
 *
 * @data [in]: pointer to the start of the data
 * @data_end [in]: pointer to the end of the data
 * @ctx [out]: pointer to the parser context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t tunnel_parse(uint8_t *data, uint8_t *data_end, struct tun_parser_ctx *ctx);

/*
 * Parse a packet and establish its direction
 *
 * @data [in]: pointer to the start of the data
 * @data_end [in]: pointer to the end of the data
 * @ctx [out]: pointer to the parser context
 * @parsed_pkt_type [out]: pointer store the incoming packet inferred type
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t unknown_parse(uint8_t *data,
			   uint8_t *data_end,
			   struct tun_parser_ctx *ctx,
			   enum parser_pkt_type *parsed_pkt_type);

#endif /* PACKET_PARSER_H_ */
