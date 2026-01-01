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

#include <stdio.h>

#include <doca_argp.h>
#include <doca_log.h>

#include "stream_receive_perf_core.h"

DOCA_LOG_REGISTER(STREAM_RECEIVE_PERF);

/*
 * Stream Receive application main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int main(int argc, char **argv)
{
	struct app_config config;
	doca_error_t ret;
	struct doca_log_backend *sdk_log;
	int exit_code = EXIT_SUCCESS;

	/* Create a logger backend that prints to the standard output */
	ret = doca_log_backend_create_standard();
	if (ret != DOCA_SUCCESS) {
		fprintf(stderr, "Logger initialization failed: %s\n", doca_error_get_name(ret));
		return EXIT_FAILURE;
	}

	/* Register a logger backend for internal SDK errors and warnings */
	ret = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (ret != DOCA_SUCCESS)
		return EXIT_FAILURE;
	ret = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (ret != DOCA_SUCCESS)
		return EXIT_FAILURE;

	if (!init_config(&config))
		return EXIT_FAILURE;
	ret = doca_argp_init(APP_NAME, &config);
	if (ret != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_name(ret));
		exit_code = EXIT_FAILURE;
		goto cleanup_config;
	}
	if (!register_argp_params()) {
		exit_code = EXIT_FAILURE;
		goto cleanup_argp;
	}
	ret = doca_argp_start(argc, argv);
	if (ret != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse application command line: %s", doca_error_get_name(ret));
		exit_code = EXIT_FAILURE;
		goto cleanup_argp;
	}

	if (config.list) {
		ret = doca_rmax_init();
		if (ret != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to initialize DOCA RMAX: %s", doca_error_get_name(ret));
			exit_code = EXIT_FAILURE;
			goto cleanup_argp;
		}

		list_devices();
	} else {
		struct globals globals;
		struct stream_data data;
		struct doca_dev *dev = NULL;

		if (!mandatory_args_set(&config)) {
			DOCA_LOG_ERR("Not all mandatory arguments were set");
			doca_argp_usage();
			exit_code = EXIT_FAILURE;
			goto cleanup_argp;
		}

		if (config.affinity_mask_set) {
			ret = doca_rmax_set_cpu_affinity_mask(config.affinity_mask);
			if (ret != DOCA_SUCCESS) {
				DOCA_LOG_ERR("Error setting CPU affinity mask: %s", doca_error_get_name(ret));
				exit_code = EXIT_FAILURE;
				goto cleanup_argp;
			}
		}
		ret = doca_rmax_init();
		if (ret != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to initialize DOCA RMAX: %s", doca_error_get_name(ret));
			exit_code = EXIT_FAILURE;
			goto cleanup_argp;
		}

		dev = open_device(&config.dev_ip);
		if (dev == NULL) {
			DOCA_LOG_ERR("Error opening device");
			exit_code = EXIT_FAILURE;
			goto cleanup_rmax;
		}

		ret = init_globals(&config, dev, &globals);
		if (ret != DOCA_SUCCESS) {
			exit_code = EXIT_FAILURE;
			goto cleanup_device;
		}

		ret = init_stream(&config, dev, &globals, &data);
		if (ret != DOCA_SUCCESS) {
			exit_code = EXIT_FAILURE;
			goto cleanup_globals;
		}

		/* main loop */
		if (!run_recv_loop(&config, &globals, &data))
			exit_code = EXIT_FAILURE;

		if (!destroy_stream(dev, &globals, &data))
			exit_code = EXIT_FAILURE;
cleanup_globals:
		if (!destroy_globals(&globals, dev))
			exit_code = EXIT_FAILURE;
cleanup_device:
		ret = doca_dev_close(dev);
		if (ret != DOCA_SUCCESS) {
			exit_code = EXIT_FAILURE;
			DOCA_LOG_ERR("Error closing device: %s", doca_error_get_name(ret));
		}
	}

cleanup_rmax:
	ret = doca_rmax_release();
	if (ret != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to release DOCA RMAX: %s", doca_error_get_name(ret));
		exit_code = EXIT_FAILURE;
	}
cleanup_argp:
	doca_argp_destroy();
cleanup_config:
	destroy_config(&config);

	return exit_code;
}
