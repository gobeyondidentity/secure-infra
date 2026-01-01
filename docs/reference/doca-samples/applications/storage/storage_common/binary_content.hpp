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

#ifndef APPLICATIONS_STORAGE_STORAGE_COMMON_BINARY_CONTENT_HPP_
#define APPLICATIONS_STORAGE_STORAGE_COMMON_BINARY_CONTENT_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace storage {

/*
 * Structure representing the content of a storage binary content (.sbc) file
 */
struct binary_content {
	uint32_t block_size;
	uint32_t block_count;
	std::vector<uint8_t> content;
};

bool file_has_binary_content_header(std::string const &file_name);

/*
 * Load a .sbc file from disk
 *
 * @file_name [in]: Name of file to load from
 * @return: loaded content
 * @throws storage::runtime_error - an error occurred
 */
storage::binary_content load_binary_content_from_file(std::string const &file_name);

/*
 * Write content into a .sbc file on disk
 *
 * @file_name [in]: Name of file to write into
 * @sbc: Content to write
 * @throws storage::runtime_error - an error occurred
 */
void write_binary_content_to_file(std::string const &file_name, storage::binary_content const &sbc);

} // namespace storage

#endif /* APPLICATIONS_STORAGE_STORAGE_COMMON_BINARY_CONTENT_HPP_ */
