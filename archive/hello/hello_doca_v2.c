/*
 * DOCA Hello World - Simple Device Enumeration
 * Compatible with DOCA 3.2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <doca_dev.h>
#include <doca_error.h>

int main(void)
{
	doca_error_t result;
	struct doca_devinfo **dev_list;
	uint32_t nb_devs = 0;
	uint32_t i;

	printf("=== DOCA Hello World ===\n");
	printf("Enumerating DOCA devices...\n\n");

	/* Get list of all available DOCA devices */
	result = doca_devinfo_create_list(&dev_list, &nb_devs);
	if (result != DOCA_SUCCESS) {
		fprintf(stderr, "Failed to create device list: %s\n", 
			doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	printf("Found %u DOCA device(s)\n\n", nb_devs);

	if (nb_devs == 0) {
		printf("No DOCA devices found\n");
		doca_devinfo_destroy_list(dev_list);
		return EXIT_SUCCESS;
	}

	/* Iterate through devices and print information */
	for (i = 0; i < nb_devs; i++) {
		char pci_addr[DOCA_DEVINFO_PCI_ADDR_SIZE] = {0};
		char iface_name[DOCA_DEVINFO_IFACE_NAME_SIZE] = {0};

		printf("Device #%u:\n", i);

		/* Get PCI address */
		result = doca_devinfo_get_pci_addr_str(dev_list[i], pci_addr);
		if (result == DOCA_SUCCESS) {
			printf("  PCI Address: %s\n", pci_addr);
		}

		/* Get interface name */
		result = doca_devinfo_get_iface_name(dev_list[i], iface_name, 
						     sizeof(iface_name));
		if (result == DOCA_SUCCESS) {
			printf("  Interface: %s\n", iface_name);
		}

		/* Try to open the device */
		struct doca_dev *dev = NULL;
		result = doca_dev_open(dev_list[i], &dev);
		if (result == DOCA_SUCCESS) {
			printf("  Status: ✓ Successfully opened\n");
			doca_dev_close(dev);
		} else {
			printf("  Status: ✗ Failed to open (%s)\n", 
			       doca_error_get_descr(result));
		}
		printf("\n");
	}

	printf("=== Device enumeration complete ===\n");

	/* Cleanup */
	doca_devinfo_destroy_list(dev_list);

	return EXIT_SUCCESS;
}
