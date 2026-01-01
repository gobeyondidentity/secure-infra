/*
 * Copyright (c) 2024 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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

#ifndef APPLICATIONS_STORAGE_STORAGE_COMMON_DEFINITIONS_HPP_
#define APPLICATIONS_STORAGE_STORAGE_COMMON_DEFINITIONS_HPP_

#include <stdexcept>
#include <string>

#include <doca_error.h>

namespace storage {

/*
 * The size of a CPU cache line. This value is typically right but it can be modified if the given platform has a
 * different cache line size
 */
uint32_t constexpr cache_line_size = 64;

/*
 * Maximum number of supported control messages
 */
uint32_t constexpr max_concurrent_control_messages = 8;

/*
 * Header data placed before the data of a compressed block
 */
struct compressed_block_header {
	uint32_t original_size;
	uint32_t compressed_size;
};

/*
 * Trailer data placed after the data of a compressed block
 */
struct compressed_block_trailer {
	uint32_t reserved;
};

/*
 *
 */
class runtime_error : public std::runtime_error {
public:
	~runtime_error() override = default;
	[[maybe_unused]] runtime_error(doca_error_t error, std::string const &msg)
		: std::runtime_error{msg},
		  m_err{error}
	{
	}
	[[maybe_unused]] runtime_error(doca_error_t error, char const *msg) : std::runtime_error{msg}, m_err{error}
	{
	}
	runtime_error(runtime_error const &) = default;
	runtime_error(runtime_error &&) noexcept = default;
	runtime_error &operator=(runtime_error const &) = default;
	runtime_error &operator=(runtime_error &&) noexcept = default;

	[[nodiscard]] doca_error_t get_doca_error() const noexcept
	{
		return m_err;
	}

private:
	doca_error_t m_err;
};

} /* namespace storage */

#endif /* APPLICATIONS_STORAGE_STORAGE_COMMON_DEFINITIONS_HPP_ */
