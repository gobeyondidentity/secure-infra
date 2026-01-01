/*
 * DOCA Crypto Hello World - SHA-256 Hash Example
 *
 * Demonstrates hardware-accelerated SHA-256 hashing using DOCA SDK.
 * This example hashes the string "Hello, DOCA Crypto!" using the hardware.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_mmap.h>
#include <doca_sha.h>
#include <doca_pe.h>
#include <doca_error.h>
#include <doca_log.h>

DOCA_LOG_REGISTER(CRYPTO_HELLO);

#define SHA256_HASH_SIZE 32
#define MAX_BUFS 2
#define SLEEP_MICROSECONDS 10

struct crypto_resources {
    struct doca_dev *dev;
    struct doca_pe *pe;
    struct doca_mmap *mmap;
    struct doca_buf_inventory *buf_inv;
    struct doca_sha *sha_ctx;
    struct doca_ctx *ctx;
    doca_error_t task_result;
    int tasks_remaining;
};

/*
 * SHA hash task completed callback
 */
static void sha_completed_callback(struct doca_sha_task_hash *task,
                                    union doca_data task_user_data,
                                    union doca_data ctx_user_data)
{
    struct crypto_resources *resources = (struct crypto_resources *)ctx_user_data.ptr;

    resources->task_result = DOCA_SUCCESS;
    resources->tasks_remaining--;

    DOCA_LOG_INFO("SHA-256 hash completed successfully");

    doca_task_free(doca_sha_task_hash_as_task(task));

    if (resources->tasks_remaining == 0)
        doca_ctx_stop(resources->ctx);
}

/*
 * SHA hash task error callback
 */
static void sha_error_callback(struct doca_sha_task_hash *task,
                                union doca_data task_user_data,
                                union doca_data ctx_user_data)
{
    struct crypto_resources *resources = (struct crypto_resources *)ctx_user_data.ptr;
    struct doca_task *doca_task = doca_sha_task_hash_as_task(task);

    resources->task_result = doca_task_get_status(doca_task);
    resources->tasks_remaining--;

    DOCA_LOG_ERR("SHA-256 hash failed: %s", doca_error_get_descr(resources->task_result));

    doca_task_free(doca_task);

    if (resources->tasks_remaining == 0)
        doca_ctx_stop(resources->ctx);
}

/*
 * Check if device supports SHA-256 hash
 */
static doca_error_t sha_is_supported(struct doca_devinfo *devinfo)
{
    return doca_sha_cap_task_hash_get_supported(devinfo, DOCA_SHA_ALGORITHM_SHA256);
}

/*
 * Print hash in hexadecimal format
 */
static void print_hash(const uint8_t *hash, size_t len)
{
    printf("SHA-256: ");
    for (size_t i = 0; i < len; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

/*
 * Initialize DOCA crypto resources
 */
static doca_error_t init_crypto_resources(struct crypto_resources *resources)
{
    doca_error_t result;
    struct doca_devinfo **dev_list;
    uint32_t nb_devs = 0;
    union doca_data ctx_user_data = {0};

    memset(resources, 0, sizeof(*resources));

    /* Find device that supports SHA */
    DOCA_LOG_INFO("Searching for device with SHA-256 support...");

    result = doca_devinfo_create_list(&dev_list, &nb_devs);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to enumerate devices: %s", doca_error_get_descr(result));
        return result;
    }

    for (uint32_t i = 0; i < nb_devs; i++) {
        if (sha_is_supported(dev_list[i]) == DOCA_SUCCESS) {
            result = doca_dev_open(dev_list[i], &resources->dev);
            if (result == DOCA_SUCCESS) {
                DOCA_LOG_INFO("Found compatible device");
                break;
            }
        }
    }

    doca_devinfo_destroy_list(dev_list);

    if (resources->dev == NULL) {
        DOCA_LOG_ERR("No device found with SHA-256 support");
        return DOCA_ERROR_NOT_FOUND;
    }

    /* Create SHA context */
    result = doca_sha_create(resources->dev, &resources->sha_ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create SHA context: %s", doca_error_get_descr(result));
        return result;
    }

    resources->ctx = doca_sha_as_ctx(resources->sha_ctx);

    /* Create progress engine */
    result = doca_pe_create(&resources->pe);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create progress engine: %s", doca_error_get_descr(result));
        return result;
    }

    /* Create memory map */
    result = doca_mmap_create(&resources->mmap);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create mmap: %s", doca_error_get_descr(result));
        return result;
    }

    result = doca_mmap_add_dev(resources->mmap, resources->dev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to add device to mmap: %s", doca_error_get_descr(result));
        return result;
    }

    /* Create buffer inventory */
    result = doca_buf_inventory_create(MAX_BUFS, &resources->buf_inv);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create buffer inventory: %s", doca_error_get_descr(result));
        return result;
    }

    result = doca_buf_inventory_start(resources->buf_inv);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start buffer inventory: %s", doca_error_get_descr(result));
        return result;
    }

    /* Register callbacks */
    result = doca_sha_task_hash_set_conf(resources->sha_ctx,
                                          sha_completed_callback,
                                          sha_error_callback,
                                          MAX_BUFS);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set SHA callbacks: %s", doca_error_get_descr(result));
        return result;
    }

    ctx_user_data.ptr = resources;
    doca_ctx_set_user_data(resources->ctx, ctx_user_data);

    /* Connect context to progress engine */
    result = doca_pe_connect_ctx(resources->pe, resources->ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to connect context to PE: %s", doca_error_get_descr(result));
        return result;
    }

    /* Start context */
    result = doca_ctx_start(resources->ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start context: %s", doca_error_get_descr(result));
        return result;
    }

    DOCA_LOG_INFO("DOCA Crypto resources initialized successfully");
    return DOCA_SUCCESS;
}

/*
 * Cleanup resources
 */
static void cleanup_crypto_resources(struct crypto_resources *resources)
{
    if (resources->ctx != NULL)
        doca_ctx_stop(resources->ctx);

    if (resources->buf_inv != NULL) {
        doca_buf_inventory_stop(resources->buf_inv);
        doca_buf_inventory_destroy(resources->buf_inv);
    }

    if (resources->mmap != NULL)
        doca_mmap_destroy(resources->mmap);

    if (resources->pe != NULL)
        doca_pe_destroy(resources->pe);

    if (resources->sha_ctx != NULL)
        doca_sha_destroy(resources->sha_ctx);

    if (resources->dev != NULL)
        doca_dev_close(resources->dev);
}

/*
 * Hash a message using hardware-accelerated SHA-256
 */
static doca_error_t hash_message(struct crypto_resources *resources,
                                  const char *message,
                                  uint8_t *hash_output)
{
    doca_error_t result;
    struct doca_buf *src_buf = NULL;
    struct doca_buf *dst_buf = NULL;
    struct doca_sha_task_hash *hash_task = NULL;
    union doca_data task_user_data = {0};
    void *src_data;
    void *dst_data;
    size_t msg_len = strlen(message);
    struct timespec sleep_time = { .tv_sec = 0, .tv_nsec = SLEEP_MICROSECONDS * 1000 };

    DOCA_LOG_INFO("Hashing message: \"%s\"", message);

    /* Allocate source buffer */
    src_data = malloc(msg_len);
    if (src_data == NULL) {
        DOCA_LOG_ERR("Failed to allocate source buffer");
        return DOCA_ERROR_NO_MEMORY;
    }
    memcpy(src_data, message, msg_len);

    /* Allocate destination buffer */
    dst_data = malloc(SHA256_HASH_SIZE);
    if (dst_data == NULL) {
        DOCA_LOG_ERR("Failed to allocate destination buffer");
        free(src_data);
        return DOCA_ERROR_NO_MEMORY;
    }

    /* Map source memory */
    result = doca_mmap_set_memrange(resources->mmap, src_data, msg_len);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to map source memory: %s", doca_error_get_descr(result));
        goto cleanup_buffers;
    }

    result = doca_mmap_start(resources->mmap);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start mmap: %s", doca_error_get_descr(result));
        goto cleanup_buffers;
    }

    /* Create DOCA buffers */
    result = doca_buf_inventory_buf_get_by_data(resources->buf_inv, resources->mmap,
                                                 src_data, msg_len, &src_buf);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create source buffer: %s", doca_error_get_descr(result));
        goto cleanup_buffers;
    }

    result = doca_buf_inventory_buf_get_by_data(resources->buf_inv, resources->mmap,
                                                 dst_data, SHA256_HASH_SIZE, &dst_buf);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create destination buffer: %s", doca_error_get_descr(result));
        goto cleanup_buffers;
    }

    /* Allocate hash task */
    result = doca_sha_task_hash_alloc_init(resources->sha_ctx, DOCA_SHA_ALGORITHM_SHA256,
                                            src_buf, dst_buf, task_user_data, &hash_task);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to allocate hash task: %s", doca_error_get_descr(result));
        goto cleanup_buffers;
    }

    /* Submit task */
    resources->tasks_remaining = 1;
    resources->task_result = DOCA_ERROR_IN_PROGRESS;

    result = doca_task_submit(doca_sha_task_hash_as_task(hash_task));
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to submit hash task: %s", doca_error_get_descr(result));
        doca_task_free(doca_sha_task_hash_as_task(hash_task));
        goto cleanup_buffers;
    }

    /* Wait for completion */
    DOCA_LOG_INFO("Waiting for hash computation...");
    while (resources->tasks_remaining > 0) {
        doca_pe_progress(resources->pe);
        nanosleep(&sleep_time, NULL);
    }

    if (resources->task_result != DOCA_SUCCESS) {
        result = resources->task_result;
        goto cleanup_buffers;
    }

    /* Copy result */
    memcpy(hash_output, dst_data, SHA256_HASH_SIZE);
    result = DOCA_SUCCESS;

cleanup_buffers:
    if (src_buf != NULL)
        doca_buf_dec_refcount(src_buf, NULL);
    if (dst_buf != NULL)
        doca_buf_dec_refcount(dst_buf, NULL);

    doca_mmap_stop(resources->mmap);
    free(src_data);
    free(dst_data);

    return result;
}

/*
 * Main function
 */
int main(void)
{
    struct crypto_resources resources;
    uint8_t hash[SHA256_HASH_SIZE];
    doca_error_t result;
    const char *message = "Hello, DOCA Crypto!";

    printf("=================================\n");
    printf("  DOCA Crypto Hello World       \n");
    printf("  SHA-256 Hardware Acceleration  \n");
    printf("=================================\n\n");

    /* Initialize */
    result = init_crypto_resources(&resources);
    if (result != DOCA_SUCCESS) {
        fprintf(stderr, "Failed to initialize crypto resources\n");
        return EXIT_FAILURE;
    }

    /* Hash the message */
    result = hash_message(&resources, message, hash);
    if (result != DOCA_SUCCESS) {
        fprintf(stderr, "Failed to hash message\n");
        cleanup_crypto_resources(&resources);
        return EXIT_FAILURE;
    }

    /* Print results */
    printf("\nMessage: \"%s\"\n", message);
    print_hash(hash, SHA256_HASH_SIZE);

    /* Cleanup */
    cleanup_crypto_resources(&resources);

    printf("\n=================================\n");
    printf("DOCA Crypto Hello World Complete!\n");
    printf("=================================\n");

    return EXIT_SUCCESS;
}
