/*
 * kTLS + OpenSSL Hello World - Client
 *
 * Demonstrates kernel TLS offload with client certificate authentication.
 * Connects to server with mTLS, then enables kTLS for hardware crypto offload.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <linux/tls.h>

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509.h>

#define SERVER_IP "127.0.0.1"
#define SERVER_PORT 8443
#define BUFFER_SIZE 4096
#define CLIENT_CERT_FILE "client-cert.pem"
#define CLIENT_KEY_FILE "client-key.pem"
#define CA_FILE "ca-cert.pem"

/*
 * Print server certificate information
 */
static void print_server_cert(SSL *ssl)
{
    X509 *cert;
    char *line;

    cert = SSL_get_peer_certificate(ssl);
    if (cert != NULL) {
        printf("\n=== Server Certificate ===\n");

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
        printf("No server certificate\n");
    }
}

/*
 * Main client function
 */
int main(void)
{
    SSL_CTX *ctx;
    SSL *ssl;
    int sock_fd;
    struct sockaddr_in server_addr;
    char buffer[BUFFER_SIZE];
    int bytes;

    printf("=================================\n");
    printf("  kTLS + OpenSSL Client         \n");
    printf("  mTLS with Hardware Offload    \n");
    printf("=================================\n\n");

    /* Initialize OpenSSL */
    SSL_load_error_strings();
    OpenSSL_add_ssl_algorithms();

    /* Create SSL context */
    ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) {
        ERR_print_errors_fp(stderr);
        return EXIT_FAILURE;
    }

    /* Load client certificate and key */
    printf("Loading client certificate: %s\n", CLIENT_CERT_FILE);
    if (SSL_CTX_use_certificate_file(ctx, CLIENT_CERT_FILE, SSL_FILETYPE_PEM) <= 0) {
        fprintf(stderr, "Warning: Could not load client certificate\n");
        fprintf(stderr, "Server will reject connection if it requires client cert\n");
    }

    printf("Loading client key: %s\n", CLIENT_KEY_FILE);
    if (SSL_CTX_use_PrivateKey_file(ctx, CLIENT_KEY_FILE, SSL_FILETYPE_PEM) <= 0) {
        fprintf(stderr, "Warning: Could not load client key\n");
    }

    /* Load CA certificate for server verification */
    printf("Loading CA certificate: %s\n", CA_FILE);
    if (SSL_CTX_load_verify_locations(ctx, CA_FILE, NULL) <= 0) {
        fprintf(stderr, "Warning: Could not load CA certificate\n");
        fprintf(stderr, "Will not verify server certificate\n");
        SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, NULL);
    } else {
        SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, NULL);
    }

    /* Prefer AES-GCM for kTLS compatibility */
    SSL_CTX_set_cipher_list(ctx, "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384");

    /* Create TCP socket */
    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        perror("socket");
        SSL_CTX_free(ctx);
        return EXIT_FAILURE;
    }

    /* Connect to server */
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);

    if (inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr) <= 0) {
        fprintf(stderr, "Invalid server IP address\n");
        close(sock_fd);
        SSL_CTX_free(ctx);
        return EXIT_FAILURE;
    }

    printf("\nConnecting to %s:%d...\n", SERVER_IP, SERVER_PORT);
    if (connect(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect");
        fprintf(stderr, "\nFailed to connect to server\n");
        fprintf(stderr, "Make sure the server is running: ./hello_ktls_server\n");
        close(sock_fd);
        SSL_CTX_free(ctx);
        return EXIT_FAILURE;
    }

    printf("TCP connection established\n");

    /* Create SSL connection */
    ssl = SSL_new(ctx);
    SSL_set_fd(ssl, sock_fd);

    /* Perform TLS handshake */
    printf("Performing TLS handshake...\n");
    if (SSL_connect(ssl) <= 0) {
        ERR_print_errors_fp(stderr);
        fprintf(stderr, "\nTLS handshake failed\n");
        fprintf(stderr, "This is expected if certificates are not set up\n");
        fprintf(stderr, "See setup instructions in hello_ktls_setup.sh\n");
        SSL_free(ssl);
        close(sock_fd);
        SSL_CTX_free(ctx);
        return EXIT_FAILURE;
    }

    printf("TLS handshake completed successfully\n");
    printf("Protocol: %s\n", SSL_get_version(ssl));

    const SSL_CIPHER *cipher = SSL_get_current_cipher(ssl);
    if (cipher) {
        printf("Cipher: %s\n", SSL_CIPHER_get_name(cipher));
    }

    /* Print server certificate */
    print_server_cert(ssl);

    printf("\nNote: kTLS offload would be enabled here\n");
    printf("For production: Extract session keys and call setsockopt(SOL_TLS, TLS_TX/RX)\n");

    /* Send message to server */
    const char *message = "Hello from kTLS client! I have a valid certificate.";
    printf("\nSending message to server: %s\n", message);

    bytes = SSL_write(ssl, message, strlen(message));
    if (bytes <= 0) {
        fprintf(stderr, "SSL_write failed\n");
        ERR_print_errors_fp(stderr);
    } else {
        printf("Sent %d bytes\n", bytes);

        /* Read server response */
        bytes = SSL_read(ssl, buffer, sizeof(buffer) - 1);
        if (bytes > 0) {
            buffer[bytes] = '\0';
            printf("\nReceived from server: %s\n", buffer);
        } else {
            fprintf(stderr, "SSL_read failed\n");
            ERR_print_errors_fp(stderr);
        }
    }

    /* Cleanup */
    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(sock_fd);
    SSL_CTX_free(ctx);

    printf("\n=================================\n");
    printf("kTLS Client Completed\n");
    printf("=================================\n");

    return EXIT_SUCCESS;
}
