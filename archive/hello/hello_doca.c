/*
 * DOCA Hello World - Device Enumeration Example
 *
 * This program demonstrates:
 * 1. DOCA device enumeration
 * 2. Querying device properties
 * 3. Basic DOCA initialization
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>

DOCA_LOG_REGISTER(HELLO_DOCA);

#define MAX_DEVICES 16

int main(int argc, char **argv)
{
	doca_error_t result;
	struct doca_devinfo **dev_list;
	uint32_t nb_devs = 0;
	uint32_t i;

	/* Initialize DOCA logging */
	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS) {
		fprintf(stderr, "Failed to create log backend: %s\n", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	DOCA_LOG_INFO("=== DOCA Hello World ===");
	DOCA_LOG_INFO("Enumerating DOCA devices...");

	/* Get list of all available DOCA devices */
	result = doca_devinfo_create_list(&dev_list, &nb_devs);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create device list: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	DOCA_LOG_INFO("Found %u DOCA device(s)", nb_devs);

	if (nb_devs == 0) {
		DOCA_LOG_WARN("No DOCA devices found");
		doca_devinfo_destroy_list(dev_list);
		return EXIT_SUCCESS;
	}

	/* Iterate through devices and print information */
	for (i = 0; i < nb_devs; i++) {
		const char *pci_addr = NULL;
		const char *dev_name = NULL;
		uint8_t is_hotplug = 0;

		DOCA_LOG_INFO("");
		DOCA_LOG_INFO("Device #%u:", i);

		/* Get PCI address */
		result = doca_devinfo_get_pci_addr_str(dev_list[i], &pci_addr);
		if (result == DOCA_SUCCESS && pci_addr != NULL) {
			DOCA_LOG_INFO("  PCI Address: %s", pci_addr);
		}

		/* Get device name */
		result = doca_devinfo_get_iface_name(dev_list[i], &dev_name);
		if (result == DOCA_SUCCESS && dev_name != NULL) {
			DOCA_LOG_INFO("  Interface Name: %s", dev_name);
		}

		/* Check if device is hotplug capable */
		result = doca_devinfo_get_is_hotplug_manager(dev_list[i], &is_hotplug);
		if (result == DOCA_SUCCESS) {
			DOCA_LOG_INFO("  Hotplug Manager: %s", is_hotplug ? "Yes" : "No");
		}

		/* Try to open the device */
		struct doca_dev *dev = NULL;
		result = doca_dev_open(dev_list[i], &dev);
		if (result == DOCA_SUCCESS) {
			DOCA_LOG_INFO("  Status: Successfully opened");

			/* Close the device */
			doca_dev_close(dev);
		} else {
			DOCA_LOG_WARN("  Status: Failed to open (%s)", doca_error_get_descr(result));
		}
	}

	DOCA_LOG_INFO("");
	DOCA_LOG_INFO("=== Device enumeration complete ===");

	/* Cleanup */
	doca_devinfo_destroy_list(dev_list);

	return EXIT_SUCCESS;
}
