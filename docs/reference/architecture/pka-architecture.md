# Bluefield PKA Architecture

## What is an OpenSSL Engine?

An **OpenSSL Engine** is a **plugin/module system** that allows OpenSSL to delegate cryptographic operations to external implementations (hardware accelerators or alternative software libraries).

### How It Works

```
┌─────────────────────────────────────────┐
│   Your Application (Rust/C/Python)     │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────▼──────────┐
        │   OpenSSL API      │  (libcrypto)
        │  (RSA_generate)    │
        └─────────┬──────────┘
                  │
     ┌────────────▼────────────┐
     │  Engine Interface       │  (dynamic plugin system)
     └────────────┬────────────┘
                  │
     ┌────────────▼────────────┐
     │   PKA Engine (pka.so)   │  (OpenSSL plugin)
     └────────────┬────────────┘
                  │
     ┌────────────▼────────────┐
     │   libPKA.so (v2.0)      │  (Native C library)
     └────────────┬────────────┘
                  │
     ┌────────────▼────────────┐
     │  mlxbf_pka kernel mod   │  (Kernel driver)
     └────────────┬────────────┘
                  │
     ┌────────────▼────────────┐
     │  PKA Hardware (ASIC)    │  (EIP-154 based)
     │  - 96 hardware rings    │
     │  - Modular arithmetic   │
     │  - RSA/ECC acceleration │
     └─────────────────────────┘
```

## Bluefield PKA Hardware Capabilities

The **PKA (Public Key Accelerator)** is based on **EIP-154** specification and provides:

### Hardware Features
- **96 Independent Hardware Rings** (`/dev/pka/0` through `/dev/pka/95`)
- **Asynchronous operation** (submit job, get result later)
- **Multi-threaded** (each thread can have its own handle)
- **DMA-based** (zero-copy operations)

### Supported Operations

**Basic Arithmetic:**
- Modular addition
- Modular subtraction  
- Modular multiplication
- Modular exponentiation
- Modular inversion

**RSA Operations:**
- RSA encryption/decryption
- RSA signature generation/verification
- RSA-CRT (Chinese Remainder Theorem) optimization
- Key sizes: 1024, 2048, 3072, 4096 bits

**Elliptic Curve Operations:**
- ECC point addition
- ECC point multiplication
- ECDSA signature generation
- ECDSA signature verification
- Curves: P-256, P-384, P-521

**Other:**
- Diffie-Hellman key exchange
- DSA (Digital Signature Algorithm)
- Cryptographically Secure RNG (CSRNG)

## Implementation Layers

### Layer 1: Kernel Module (`mlxbf_pka`)
- **Location**: `/lib/modules/.../mlxbf-pka.ko`
- **Purpose**: Hardware device driver
- **Interface**: Creates `/dev/pka/*` character devices
- **Operations**: Memory mapping, DMA, interrupt handling

### Layer 2: Native C Library (`libPKA.so`)
- **Location**: `/usr/lib/aarch64-linux-gnu/libPKA.so`
- **Headers**: `/usr/include/pka*.h`
- **Purpose**: Userspace API to PKA hardware
- **Key Functions**:
  - `pka_init_global()` - Initialize PKA instance
  - `pka_init_local()` - Get thread handle
  - `pka_rsa()` - RSA operation
  - `pka_ecdsa_signature_generate()` - ECDSA signing
  - `pka_get_rslt()` - Get async results
  - `pka_term_local()` / `pka_term_global()` - Cleanup

### Layer 3: OpenSSL Engine (`pka.so`)
- **Location**: `/usr/lib/aarch64-linux-gnu/engines-3/pka.so`
- **Purpose**: OpenSSL plugin wrapper around libPKA
- **Advantage**: Works with existing OpenSSL-based code
- **Limitation**: Adds abstraction layer overhead

## How to Implement in Rust (3 Options)

### Option 1: OpenSSL CLI (Current Implementation)
**Pros**: Simple, no FFI, works today  
**Cons**: Process overhead, no control over async operations

```rust
Command::new("openssl")
    .args(&["genrsa", "-engine", "pka", "-out", "key.pem", "2048"])
    .output()?;
```

### Option 2: Direct FFI to libPKA (Native)
**Pros**: Direct hardware access, full async control, best performance  
**Cons**: Requires FFI bindings, more complex

```rust
#[link(name = "PKA")]
extern "C" {
    fn pka_init_global(name: *const c_char, flags: u64, ...) -> u64;
    fn pka_rsa(handle: *mut c_void, ...) -> c_int;
}
```

### Option 3: Rust OpenSSL crate with Engine API
**Pros**: Type-safe Rust, leverages openssl crate  
**Cons**: Engine API not exposed in current openssl crate version

## Recommendation for Beyond Identity Integration

For your **Pattern A (mTLS)** implementation, I recommend:

**For Key Generation**: Use **Option 1** (OpenSSL CLI)
- Simple, reliable, works today
- Key generation is not latency-critical (done once during enrollment)
- Process overhead is acceptable for infrequent operation

**For Crypto Operations (sign/verify)**: Use **Option 2** (Direct FFI)
- TLS handshake performance matters
- Async operation support needed for high throughput
- Full control over PKA ring allocation

I can create a complete Rust FFI implementation if you want to use the native libPKA API directly.

## Performance Characteristics

**Hardware-Accelerated (PKA):**
- RSA-2048 keygen: ~300ms
- ECDSA P-256 keygen: ~9ms  
- RSA-2048 sign: ~1-2ms
- ECDSA P-256 sign: ~0.5ms

**Software (ARM CPU):**
- RSA-2048 keygen: ~2-5 seconds
- ECDSA P-256 keygen: ~50-100ms
- RSA-2048 sign: ~10-20ms
- ECDSA P-256 sign: ~2-5ms

**Speedup**: 10-20x faster with PKA hardware acceleration
