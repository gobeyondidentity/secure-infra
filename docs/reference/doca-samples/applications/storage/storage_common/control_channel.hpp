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

#ifndef APPLICATIONS_STORAGE_STORAGE_COMMON_CONTROL_CHANNEL_HPP_
#define APPLICATIONS_STORAGE_STORAGE_COMMON_CONTROL_CHANNEL_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <doca_dev.h>
#include <doca_comch.h>

#include <storage_common/control_message.hpp>
#include <storage_common/ip_address.hpp>

namespace storage::control {

class channel {
public:
	virtual ~channel() = default;

	virtual bool is_connected() = 0;
	virtual void send_message(message const &msg) = 0;
	virtual storage::control::message *poll() = 0;
};

class comch_channel : public storage::control::channel {
public:
	virtual doca_comch_connection *get_comch_connection() const noexcept = 0;

	using consumer_event_callback = std::function<void(void *user_data, uint32_t consumer_id)>;
};

std::unique_ptr<storage::control::comch_channel> make_comch_client_control_channel(
	doca_dev *dev,
	char const *channel_name,
	void *callback_user_data,
	comch_channel::consumer_event_callback new_consumer_event_cb,
	comch_channel::consumer_event_callback expired_consumer_event_cb);

std::unique_ptr<storage::control::comch_channel> make_comch_server_control_channel(
	doca_dev *dev,
	doca_dev_rep *dev_rep,
	char const *channel_name,
	void *callback_user_data,
	comch_channel::consumer_event_callback new_consumer_event_cb,
	comch_channel::consumer_event_callback expired_consumer_event_cb);

std::unique_ptr<storage::control::channel> make_tcp_client_control_channel(storage::ip_address const &server_address);

std::unique_ptr<storage::control::channel> make_tcp_server_control_channel(uint16_t listen_port);

} // namespace storage::control

#endif /* APPLICATIONS_STORAGE_STORAGE_COMMON_CONTROL_CHANNEL_HPP_ */
