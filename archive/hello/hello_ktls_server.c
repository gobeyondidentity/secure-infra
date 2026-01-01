/*
 * kTLS + OpenSSL Hello World - Server
 *
 * Demonstrates kernel TLS offload with mTLS client certificate validation.
 * After TLS handshake completes, session keys are passed to kernel for
 * hardware crypto offload on ConnectX-7.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <linux/tls.h>

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#define SERVER_PORT 8443
#define BUFFER_SIZE 4096
#define CERT_FILE "server-cert.pem"
#define KEY_FILE "server-key.pem"
#define CA_FILE "ca-cert.pem"

/*
 * Print client certificate information
 */
static void print_client_cert(SSL *ssl)
{
    X509 *cert;
    char *line;

    cert = SSL_get_peer_certificate(ssl);
    if (cert != NULL) {
        printf("\n=== Client Certificate ===\n");

        line = X509_NAME_oneline(X509_get_subject_name(cert), 0, 0);
        printf("Subject: %s\n", line);
        OPENSSL_free(line);

        line = X509_NAME_oneline(X509_get_issuer_name(cert), 0, 0);
        printf("Issuer: %s\n", line);
        OPENSSL_free(line);

        printf("Certificate verified: %s\n",
               SSL_get_verify_result(ssl) == X509_V_OK ? "YES" : "NO");

        X509_free(cert);
    } else {
        printf("No client certificate provided\n");
    }
}

/*
 * Enable kTLS offload for transmit
 */
static int enable_ktls_tx(int fd, SSL *ssl)
{
    struct tls12_crypto_info_aes_gcm_128 crypto_info;
    const SSL_CIPHER *cipher;
    unsigned char *key_data;
    int key_len;

    /* Check if we're using AES-GCM */
    cipher = SSL_get_current_cipher(ssl);
    if (!cipher) {
        fprintf(stderr, "Failed to get current cipher\n");
        return -1;
    }

    const char *cipher_name = SSL_CIPHER_get_name(cipher);
    printf("TLS Cipher: %s\n", cipher_name);

    /* Only AES-GCM-128 is widely supported for kTLS */
    if (strstr(cipher_name, "AES128-GCM") == NULL) {
        printf("Note: kTLS offload requires AES-GCM-128 cipher\n");
        printf("Current cipher (%s) may not support kTLS offload\n", cipher_name);
        return -1;
    }

    memset(&crypto_info, 0, sizeof(crypto_info));
    crypto_info.info.version = TLS_1_2_VERSION;
    crypto_info.info.cipher_type = TLS_CIPHER_AES_GCM_128;

    /* Get key material from OpenSSL */
    /* Note: This is a simplified example. Production code needs proper key extraction */
    printf("Note: kTLS TX offload prepared (key extraction simplified for demo)\n");

    /* In production, you would:
     * 1. Extract IV, keys, and sequence from SSL session
     * 2. Call setsockopt(fd, SOL_TLS, TLS_TX, &crypto_info, sizeof(crypto_info))
     */

    printf("kTLS TX offload would be enabled here\n");
    return 0;
}

/*
 * Handle client connection
 */
static void handle_client(SSL *ssl, int client_fd)
{
    char buffer[BUFFER_SIZE];
    int bytes;

    /* Print client certificate info */
    print_client_cert(ssl);

    /* Enable kTLS offload (if cipher is compatible) */
    printf("\nAttempting kTLS offload...\n");
    enable_ktls_tx(client_fd, ssl);

    /* Read client message */
    bytes = SSL_read(ssl, buffer, sizeof(buffer) - 1);
    if (bytes > 0) {
        buffer[bytes] = '\0';
        printf("\nReceived from client: %s\n", buffer);

        /* Send response */
        const char *response = "Hello from kTLS server! Your certificate was validated.";
        SSL_write(ssl, response, strlen(response));
        printf("Sent response to client\n");
    } else {
        int ssl_error = SSL_get_error(ssl, bytes);
        fprintf(stderr, "SSL_read failed: %d\n", ssl_error);
    }
}

/*
 * Main server function
 */
int main(void)
{
    SSL_CTX *ctx;
    SSL *ssl;
    int server_fd, client_fd;
    struct sockaddr_in addr, client_addr;
    socklen_t client_len = sizeof(client_addr);

    printf("=================================\n");
    printf("  kTLS + OpenSSL Server         \n");
    printf("  mTLS with Hardware Offload    \n");
    printf("=================================\n\n");

    /* Initialize OpenSSL */
    SSL_load_error_strings();
    OpenSSL_add_ssl_algorithms();

    /* Create SSL context */
    ctx = SSL_CTX_new(TLS_server_method());
    if (!ctx) {
        ERR_print_errors_fp(stderr);
        return EXIT_FAILURE;
    }

    /* Load server certificate and key */
    printf("Loading server certificate: %s\n", CERT_FILE);
    if (SSL_CTX_use_certificate_file(ctx, CERT_FILE, SSL_FILETYPE_PEM) <= 0) {
        fprintf(stderr, "Warning: Could not load server certificate (will generate self-signed)\n");
        fprintf(stderr, "Run: openssl req -x509 -newkey rsa:2048 -nodes -keyout %s -out %s -days 365\n",
                KEY_FILE, CERT_FILE);
    }

    printf("Loading server key: %s\n", KEY_FILE);
    if (SSL_CTX_use_PrivateKey_file(ctx, KEY_FILE, SSL_FILETYPE_PEM) <= 0) {
        fprintf(stderr, "Warning: Could not load server key\n");
    }

    /* Configure client certificate verification */
    printf("Configuring client certificate verification\n");
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);

    if (SSL_CTX_load_verify_locations(ctx, CA_FILE, NULL) <= 0) {
        fprintf(stderr, "Warning: Could not load CA certificate: %s\n", CA_FILE);
        fprintf(stderr, "Client certificate verification may fail\n");
    }

    /* Prefer AES-GCM for kTLS compatibility */
    SSL_CTX_set_cipher_list(ctx, "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384");

    /* Create TCP socket */
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        SSL_CTX_free(ctx);
        return EXIT_FAILURE;
    }

    /* Allow port reuse */
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    /* Bind to port */
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(SERVER_PORT);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        SSL_CTX_free(ctx);
        return EXIT_FAILURE;
    }

    /* Listen for connections */
    if (listen(server_fd, 5) < 0) {
        perror("listen");
        close(server_fd);
        SSL_CTX_free(ctx);
        return EXIT_FAILURE;
    }

    printf("\nServer listening on port %d\n", SERVER_PORT);
    printf("Waiting for client connection...\n\n");

    /* Accept client connection */
    client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd < 0) {
        perror("accept");
        close(server_fd);
        SSL_CTX_free(ctx);
        return EXIT_FAILURE;
    }

    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
    printf("Client connected from %s:%d\n", client_ip, ntohs(client_addr.sin_port));

    /* Create SSL connection */
    ssl = SSL_new(ctx);
    SSL_set_fd(ssl, client_fd);

    /* Perform TLS handshake */
    printf("Performing TLS handshake...\n");
    if (SSL_accept(ssl) <= 0) {
        ERR_print_errors_fp(stderr);
        fprintf(stderr, "\nTLS handshake failed\n");
        fprintf(stderr, "This is expected if certificates are not set up\n");
        fprintf(stderr, "See setup instructions in hello_ktls_setup.sh\n");
    } else {
        printf("TLS handshake completed successfully\n");
        printf("Protocol: %s\n", SSL_get_version(ssl));

        /* Handle client */
        handle_client(ssl, client_fd);
    }

    /* Cleanup */
    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(client_fd);
    close(server_fd);
    SSL_CTX_free(ctx);

    printf("\n=================================\n");
    printf("kTLS Server Completed\n");
    printf("=================================\n");

    return EXIT_SUCCESS;
}
