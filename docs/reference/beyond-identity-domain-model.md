# Domain Model

## Overview

Platform-wide domain model spanning the entire server-side architecture.
This specification defines the canonical entity model for multi-tenant
identity management, credential lifecycle, cryptographic infrastructure,
and service integration across all Beyond Identity cloud services.

Throughout this specification, **entity** refers to a class or type
definition (e.g., `Tenant.Realm.Directory.Identity`) while **resource**
refers to a specific instance of that entity class.

The domain model represents three fundamental dimensions of the service:

**Capabilities** - System features and supported operations. Defines what
is architecturally possible within the platform (protocols, credential
types, authentication methods, token formats).

**Configuration** - Administratively controlled settings. Entities that
operators create, modify, and delete to customize service behavior
(tenants, realms, policies, directories, signing authorities).

**Operational State** - Active runtime entities requiring tracking.
System-managed records reflecting current service state (sessions,
authorizations, tokens, log entries).

**Historical State:** The append-only event log (Tenant.Log) provides
complete historical record of all configuration changes and operational
state transitions. Future optimizations may introduce specialized indices
for recent history lookups, but such implementation details remain outside
the scope of this domain model specification.

**Domain Model Principles:** The hierarchical organization of entities
expresses relationships and composition, not access control. An entity
scoped within a realm does not imply that only identities from that realm
can modify it. The domain model defines abstractions for administrative
manipulation, their relationships, and lifecycle—authorization is
determined exclusively through policy rules, scopes, and issued tokens or
API keys, not through entity hierarchy.

This model strives for completeness with no hidden assumptions or
implicit behaviors. All system behavior should be explicitly represented
in the model or clearly documented as an implementation detail. While rare
exceptions may exist, this principle guides 99.9% of the architecture.

**Documentation Standards:** See [conventions.md](conventions.md) for
complete domain model conventions (entities, relationships, cardinality,
tree markers, schema formats).

**Related Specifications:**
- [language.md](language.md) - Type system, expressions, constraints
- [schemas/](schemas/) - Detailed entity schemas

## Entity Relationships

```
BeyondIdentity (1)                                api(read)
└── Tenant (0:N)                                  api(create,read,list,delete)
    ├── Log (1:N)                                 api(read,list)
    ├── Policy [namespace]
    │   └── Rule (0:N)                            api(create,read,list,update,delete)
    ├── Analytic [namespace]
    │   ├── Inventory (0:N)                       api(read,list)
    │   ├── Running (0:N)                         api(create,read,list,delete)
    │   └── Report (0:N)                          api(read,list)
    │
    └── Realm (1:N)                               api(create,read,list,delete)
        ├── API [namespace]
        │   ├── Context (1)                       api(read)
        │   ├── Scope (0:N)                       api(create,read,list,delete)
        │   └── Key (0:N)                         api(create,read,list,delete)
        ├── Signing Authority [namespace]
        │   ├── X509 CA (0:N)                     api(create,read,list,delete)
        │   ├── JWT A (0:N)                       api(create,read,list,delete)
        │   ├── SSH A (0:N)                       api(create,read,list,delete)
        │   └── PGP A (0:N)                       api(create,read,list,delete)
        │
        ├── Directory (1:N)                       api(create,read,list,update,delete)
        │   ├── Schema (1)                        api(create,read,update,delete)
        │   ├── Source (0:1)                      api(create,read,update,delete)
        │   ├── Replica (0:N)                     api(create,read,list,update,delete)
        │   ├── Identity (0:N)                    api(create,read,list,update,delete)
        │   ├── Group (0:N)                       api(create,read,list,update,delete)
        │   └── Credential (0:N)                  api(create,read,list,delete)
        │
        └── IdP (1:N)                             api(create,read,list,update,delete)
            ├── Session (0:N)                     api(read,list)
            ├── Authorization (0:N)               api(create,read,list,delete)
            ├── Token (0:N)                       api(read,list)
            ├── Protocol [namespace]
            │   ├── OAuth2 (0:1)                  api(create,read,update,delete)
            │   ├── OIDC (0:1) [TBD]
            │   ├── SAML (0:1) [TBD]
            │   └── WS-Fed (0:1) [TBD]
            ├── Client (0:N)                      api(create,read,list,update,delete)
            │   ├── Context (1)                   api(create,read,update,delete)
            │   ├── Scopes (0:N)                  api(create,read,list,update,delete)
            │   └── Protocol [namespace]
            │       ├── OAuth2 (0:1)              api(create,read,update,delete)
            │       ├── OIDC (0:1) [TBD]
            │       ├── SAML (0:1) [TBD]
            │       └── WS-Fed (0:1) [TBD]
            └── ExtIdP (0:N)                      api(create,read,list,update,delete)
                └── Protocol [namespace]
                    ├── OAuth2 (0:1)              api(create,read,update,delete)
                    ├── OIDC (0:1) [TBD]
                    ├── SAML (0:1) [TBD]
                    └── WS-Fed (0:1) [TBD]
        │
        ├── Device (0:N)                          api(read,list)
        ├── Workloads (0:N) [TBD]
        ├── Hooks (0:N) [TBD]
        ├── Providers [namespace]
        │   ├── Log/SIEM (0:N)                    api(create,read,list,update,delete)
        │   ├── Email (0:N) [TBD]
        │   └── SMS (0:N) [TBD]
        └── Branding (1) [TBD]
```

### BeyondIdentity

Root entity representing the Beyond Identity platform. Contains domain
model metadata including the semantic version of this specification.

**Allowed Operations:**
- `read` - Retrieve platform metadata

```
BeyondIdentity
└── version: SemVer              /// domain model semantic version
```

**SemVer Type:**
```
SemVer: Type = Record
├── major: Integer           /// Breaking changes to domain model
├── minor: Integer           /// Additive, backwards-compatible changes
└── patch: Integer           /// Backwards-compatible fixes/clarifications
```

[Schema](schemas/platform.md#beyondidentity)

### Tenant

Top-level container for all administrative resources. Primary entity
in the domain model.

**Allowed Operations:**
- `create` - Provision new tenant
- `read` - Retrieve tenant by ID
- `list` - Query tenants
- `delete` - Remove tenant and all contained resources

```
Tenant
├── id: UUID                     /// system-level identifier
└── name: String                 /// user-friendly tenant name
```

[Schema](schemas/tenant.md#tenant)

### Tenant.Log

Append-only event log capturing all entity CRUD operations. Complete
historical record from tenant inception for audit, forensics, and
troubleshooting.

**Allowed Operations:**
- `read` - Retrieve log entry by ID
- `list` - Query log entries (append-only, system-generated)

```
Log
├── id: UUID                     /// system-level identifier
├── timestamp: Datetime          /// time event was logged
├── correlation_id: UUID         /// correlates related events
├── entity_id: UUID              /// affected entity
└── subsystem: Variant           /// subsystem event (tag identifies subsystem)
    | API: APIEvent
    | SA: SAEvent
    | DIRECTORY: DirectoryEvent
    | AUTH: AuthEvent
    | ANALYTIC: AnalyticEvent
    | HOOKS: HooksEvent
    | WORKFLOW: WorkflowEvent
```

[Schema](schemas/tenant.md#log)

### Tenant.Policy.Rule

ABAC-based policy rule defining dispositions for transactions or
conditions. Primary access control mechanism supporting complex
permissions and hierarchies.

**Allowed Operations:**
- `create` - Define new policy rule
- `read` - Retrieve rule by ID
- `list` - Query policy rules
- `update` - Modify rule logic or dispositions
- `delete` - Remove policy rule

```
Rule
├── id: UUID                     /// system-level identifier
├── name: String                 /// for admin reference
├── match: Expression[Boolean]   /// predicate for rule application
└── action: Variant              /// transaction disposition
    | ALLOW: AllowAction         /// allow with provisions
    | DENY: DenyAction           /// deny with information
    | IGNORE: IgnoreAction       /// silent deny
    | CHALLENGE: ChallengeAction /// retry with additional info
```

[Schema](schemas/tenant.md#rule)

### Tenant.Analytic.Inventory

System-provided catalog of all available analytics. Defines analytic
types that can be activated for tenant operations and security monitoring.
Examples: stale API keys, stale user credentials, heavy hitters (high
usage), fast movers (rapid activity), anomalous access patterns.

**Allowed Operations:**
- `read` - Retrieve analytic definition by ID
- `list` - Query available analytics (system-managed)

```
Inventory
├── id: UUID                     /// system-level identifier
├── name: String                 /// analytic identifier
├── description: String          /// what this analytic detects
├── execution_mode: ExecutionMode /// CONTINUOUS/BATCH/ON_DEMAND
├── schedule: String             /// cron (BATCH only)
├── alert_types: List[AlertType] /// alert categories
└── produces_reports: Boolean    /// generates reports
```

[Schema](schemas/tenant.md#analyticinventory)

### Tenant.Analytic.Running

Active analytics for this tenant. Administrators activate analytics from
the Inventory to monitor their tenant's operations and security posture.

**Allowed Operations:**
- `create` - Activate an analytic from Inventory
- `read` - Retrieve running analytic instance by ID
- `list` - Query active analytics
- `delete` - Deactivate and remove analytic

```
Running
├── id: UUID                     /// system-level identifier
├── inventory_id: UUID           /// reference to Analytic.Inventory
├── started_at: Datetime         /// when activated
└── state: AnalyticState         /// STARTING/RUNNING/STOPPED/FAILED
```

[Schema](schemas/tenant.md#analyticrunning)

### Tenant.Analytic.Report

Historical reports generated by batch and on-demand analytics. Provides
audit trail of analytic executions and findings.

**Allowed Operations:**
- `read` - Retrieve report by ID
- `list` - Query reports (system-generated)

```
Report
├── id: UUID                     /// system-level identifier
├── running_id: UUID             /// which instance generated this
├── executed_at: Datetime        /// when report was generated
└── report_data: JSON            /// report content
```

[Schema](schemas/tenant.md#analyticreport)

### Tenant.Realm

Administrative namespace for delegating management to sub-organizations
or compartmentalizing operations.

**Allowed Operations:**
- `create` - Provision new realm
- `read` - Retrieve realm by ID
- `list` - Query realms within tenant
- `delete` - Remove realm and all contained resources

```
Realm
├── id: UUID                     /// system-level identifier
└── name: String                 /// globally unique FQDN
                                 /// (e.g., auth.example.com)
```

[Schema](schemas/tenant.md#realm)

### Tenant.Realm.API.Context

Evaluation context available during permission constraint evaluation.
System-managed singleton defining the structure of ctx variables
referenced in permission constraints.

**Allowed Operations:**
- `read` - Retrieve context definition

```
Context
├── tenant.id: UUID              /// Tenant.id making the request
├── realm.id: UUID               /// Tenant.Realm.id
├── identity.id: UUID            /// Tenant.Realm.Directory.Identity.id
└── timestamp: Datetime          /// when the request was made
```

[Schema](schemas/api.md#context)

### Tenant.Realm.API.Scope

Authorization scope defining permissions granted to API keys. Cannot be
deleted while referenced by active or unexpired keys.

**Allowed Operations:**
- `create` - Define new scope with permissions
- `read` - Retrieve scope by ID
- `list` - Query scopes within realm
- `delete` - Remove scope (blocked if referenced by active keys)

```
Expression: Type = String
/// Boolean expression with operators:
///   Literals: true (allow all), false (deny all)
///   Relational: ==, !=, <, >, <=, >=
///   Logical: AND, OR, NOT
///   Set: in, contains
/// Context variables: ctx.tenant.id, ctx.realm.id, ctx.identity.id
/// Resource variables: resource.<attribute>
/// Functions: now()

Permission: Type = Record
├── entity: String               /// FQN of target entity class
├── description: String          /// optional human-readable description
├── operations: Set[Operation]   /// {create, read, list, update, delete}
│                                /// empty set {} allowed (grants no operations)
└── constraint: Expression       /// required; defaults to 'true' (allow all)
```

**Permission Evaluation:**

Permissions are evaluated with default-deny semantics:

1. **Matching:** Find all permissions matching the requested entity and
   operation across all scopes referenced by the API key
2. **Evaluation:** For each matching permission, evaluate its constraint:
   - Constraint evaluates to `true` → grant access
   - Constraint evaluates to `false` → skip to next permission
   - Any permission grants access → **allow**
3. **Default:** No matching permission → **deny**

**Permission Composition:**
- Multiple permissions within a scope combine as union (any grants access)
- Multiple scopes on a key combine as union (any permission from any scope
  grants access)
- Empty operations set `{}` grants no operations (useful for testing/templates)

**Scope Rendering:**

When minting a token from a Key, the referenced Scope names are concatenated
into the token's `scope` field as a space-separated string (per OAuth 2.0).
At evaluation time, the scope string is parsed, each scope is looked up by
name, and all permissions are evaluated.

```
Scope
├── id: UUID                     /// system-level identifier
├── name: String                 /// scope identifier (e.g., read:users)
├── description: String          /// optional human-readable description
└── permissions: List[Permission] /// list of permissions
```

[Schema](schemas/api.md#scope)

### Tenant.Realm.API.Key

**Allowed Operations:**
- `create` - Issue new API key
- `read` - Retrieve key by ID
- `list` - Query keys within realm
- `delete` - Revoke and remove key (immutable, cannot update)

```
BearerToken: Type = Record
├── iss: String                  /// issuer (realm FQDN)
├── sub: UUID                    /// subject (identity_id)
├── aud: String                  /// audience (API/service identifier)
├── exp: Datetime                /// expiration time
├── iat: Datetime                /// issued at time
├── nbf: Datetime                /// not before time
├── jti: UUID                    /// JWT ID (unique token identifier)
└── scope: String                /// space-separated scope values

JWTToken: Type = Record
├── iss: String                  /// issuer (realm FQDN)
├── sub: UUID                    /// subject (identity_id)
├── aud: String                  /// audience (API/service identifier)
├── exp: Datetime                /// expiration time
├── iat: Datetime                /// issued at time
├── nbf: Datetime                /// not before time
├── jti: UUID                    /// JWT ID (unique token identifier)
└── scope: String                /// space-separated scope values

DPoPToken: Type = Record
├── cnf: String                  /// public key thumbprint (RFC 7800)
├── iss: String                  /// issuer (realm FQDN)
├── sub: UUID                    /// subject (identity_id)
├── aud: String                  /// audience (API/service identifier)
├── exp: Datetime                /// expiration time
├── iat: Datetime                /// issued at time
├── nbf: Datetime                /// not before time
├── jti: UUID                    /// JWT ID (unique token identifier)
└── scope: String                /// space-separated scope values

Key
├── id: UUID                     /// system-level identifier
├── name: String                 /// for admin reference
├── description: String          /// purpose/notes for admin
└── credential: Variant          /// key type and data
    | BEARER: BearerToken        /// bearer token (RFC 6750)
    | JWT: JWTToken              /// signed JWT (RFC 7519)
    | DPOP: DPoPToken            /// DPoP-bound token (RFC 9449)
```

[Schema](schemas/api.md#key)

### Tenant.Realm.Signing_Authority.X509_CA

X.509 intermediate CA. Issues and signs leaf certificates for devices,
users, and other identity credentials. Multiple active instances
supported; freshest used for signing. Regions may maintain independent
credentials for performance and isolation.

**Allowed Operations:**
- `create` - Provision new intermediate CA
- `read` - Retrieve CA by ID
- `list` - Query CAs within realm
- `delete` - Revoke and remove CA (immutable, cannot update)

```
X509_CA
├── id: UUID                     /// system-level identifier
├── name: String                 /// for admin reference
├── parent_cert: Blob            /// parent CA certificate
├── cert: Blob                   /// this CA's certificate
├── serial_number: String        /// from cert
├── subject_dn: String           /// from cert
├── issuer_dn: String            /// from cert
├── not_before: Datetime         /// from cert
├── not_after: Datetime          /// from cert
├── algorithm: String            /// from cert
├── name_constraints: JSON       /// from cert
└── status: Enum(ACTIVE, SUSPENDED, EXPIRED)
```

[Schema](schemas/signing-authority.md#x509_ca)

### Tenant.Realm.Signing_Authority.JWT_A

X.509 leaf certificate for JWT signing. Issues and signs JSON Web
Tokens for authentication and authorization. Multiple active instances
supported; freshest used for signing. Regions may maintain independent
credentials for performance and isolation.

**Allowed Operations:**
- `create` - Provision new JWT signing certificate
- `read` - Retrieve JWT authority by ID
- `list` - Query JWT authorities within realm
- `delete` - Revoke and remove JWT authority (immutable, cannot update)

```
JWT_A
├── id: UUID                     /// system-level identifier
├── name: String                 /// for admin reference
├── x509_ca_id: UUID             /// issuing CA reference
├── cert: Blob                   /// this JWT signing certificate
├── kid: String                  /// SHA-256 thumbprint of cert (x5t#S256)
├── serial_number: String        /// from cert
├── subject_dn: String           /// from cert
├── issuer_dn: String            /// from cert
├── not_before: Datetime         /// from cert
├── not_after: Datetime          /// from cert
├── algorithm: String            /// from cert
├── public_key: Blob             /// from cert (for JWKS)
└── status: Enum(ACTIVE, SUSPENDED, EXPIRED)
```

[Schema](schemas/signing-authority.md#jwt_a)

### Tenant.Realm.Signing_Authority.SSH_A

X.509 leaf certificate for SSH CA operations. Keypair converted to SSH
format for issuing user and host SSH certificates. Multiple active
instances supported; freshest used for signing. Regions may maintain
independent credentials for performance and isolation.

**Allowed Operations:**
- `create` - Provision new SSH CA certificate
- `read` - Retrieve SSH authority by ID
- `list` - Query SSH authorities within realm
- `delete` - Revoke and remove SSH authority (immutable, cannot update)

```
SSH_A
├── id: UUID                     /// system-level identifier
├── name: String                 /// for admin reference
├── x509_ca_id: UUID             /// issuing CA reference
├── cert: Blob                   /// X.509 certificate
├── ca_type: Enum(USER, HOST)    /// user auth or host auth
├── public_key: Blob             /// SSH public key (from cert)
├── fingerprint: String          /// SHA-256 fingerprint of public_key
├── serial_number: String        /// from cert
├── subject_dn: String           /// from cert
├── issuer_dn: String            /// from cert
├── not_before: Datetime         /// from cert
├── not_after: Datetime          /// from cert
├── algorithm: String            /// from cert
└── status: Enum(ACTIVE, SUSPENDED, EXPIRED)
```

[Schema](schemas/signing-authority.md#ssh_a)

### Tenant.Realm.Signing_Authority.PGP_A

X.509 leaf certificate for PGP certification. Keypair converted to PGP
master key format for signing user PGP keys. Multiple active instances
supported; freshest used for certification. Regions may maintain
independent credentials for performance and isolation.

**Allowed Operations:**
- `create` - Provision new PGP certification authority
- `read` - Retrieve PGP authority by ID
- `list` - Query PGP authorities within realm
- `delete` - Revoke and remove PGP authority (immutable, cannot update)

```
PGP_A
├── id: UUID                     /// system-level identifier
├── name: String                 /// for admin reference
├── x509_ca_id: UUID             /// issuing CA reference
├── cert: Blob                   /// X.509 certificate
├── key_id: String               /// PGP key ID (16 hex chars)
├── fingerprint: String          /// PGP fingerprint (40 hex chars)
├── public_key: Blob             /// PGP public key (OpenPGP format)
├── serial_number: String        /// from cert
├── subject_dn: String           /// from cert
├── issuer_dn: String            /// from cert
├── not_before: Datetime         /// from cert
├── not_after: Datetime          /// from cert
├── algorithm: String            /// from cert
└── status: Enum(ACTIVE, SUSPENDED, EXPIRED)
```

[Schema](schemas/signing-authority.md#pgp_a)

### Tenant.Realm.Directory

Identity directory supporting humans, machines, drones, and other entity
types. Multiple directories enable identity partitioning for development,
debugging, or operational separation.

**Allowed Operations:**
- `create` - Provision new directory
- `read` - Retrieve directory by ID
- `list` - Query directories within realm
- `update` - Modify directory configuration
- `delete` - Remove directory and all contained identities

```
Directory
├── id: UUID                     /// system-level identifier
└── name: String                 /// unique within realm
```

[Schema](schemas/directory.md#directory)

### Tenant.Realm.Directory.Schema

Flexible identity attribute schema with minimal assumptions. Defines
globally unique ID, supported credential types, and administrator-defined
custom attributes.

**Allowed Operations:**
- `create` - Define initial schema for directory
- `read` - Retrieve schema for directory
- `update` - Modify schema definitions
- `delete` - Remove schema (only if directory empty)

```
Schema
├── id: UUID                     /// system-level identifier
└── definitions: List[String]    /// name: Type | constraint = default
```

[Schema](schemas/directory.md#directoryschema)

### Tenant.Realm.Directory.Source

Integration with upstream directory source of truth controlling identity
provisioning. Expression-based mappers filter and transform inbound data
from differing upstream schemas into directory-appropriate format.

**Allowed Operations:**
- `create` - Configure upstream source integration
- `read` - Retrieve source configuration
- `update` - Modify source mappers or settings
- `delete` - Remove source integration

```
Source
├── id: UUID                     /// system-level identifier
├── name: String                 /// for admin reference
└── mappers: List[Expression]    /// filter and transform upstream data
```

[Schema](schemas/directory.md#directorysource)

### Tenant.Realm.Directory.Replica

Integration with downstream directories or applications for identity
replication. Expression-based mappers filter and transform outbound
identity changes from local schema into downstream-appropriate format.

**Allowed Operations:**
- `create` - Configure downstream replica integration
- `read` - Retrieve replica configuration by ID
- `list` - Query configured replicas
- `update` - Modify replica mappers or settings
- `delete` - Remove replica integration

```
Replica
├── id: UUID                     /// system-level identifier
├── name: String                 /// for admin reference
└── mappers: List[Expression]    /// filter and transform outbound data
```

[Schema](schemas/directory.md#directoryreplica)

### Tenant.Realm.Directory.Identity

Identity entity binding credentials to unique identifier with
administrator-defined attributes. Flexible schema enables custom
attributes for humans, machines, drones, and other entity types.

**Allowed Operations:**
- `create` - Provision new identity
- `read` - Retrieve identity by ID
- `list` - Query identities within directory
- `update` - Modify identity attributes
- `delete` - Remove identity and revoke credentials

```
Identity
├── id: UUID                     /// system-level identifier
├── username: String             /// unique within directory
├── email: String                /// optional email address
└── credentials: List[UUID]      /// references to credentials
```

[Schema](schemas/directory.md#directoryidentity)

### Tenant.Realm.Directory.Group

Collection of identities for authorization and policy management.
Supports flexible membership rules for access control.

**Allowed Operations:**
- `create` - Define new group
- `read` - Retrieve group by ID
- `list` - Query groups within directory
- `update` - Modify group membership or attributes
- `delete` - Remove group

```
Group
├── id: UUID                     /// system-level identifier
├── name: String                 /// unique within directory
├── description: String          /// human-readable purpose
└── members: List[UUID]          /// references to identities
```

[Schema](schemas/directory.md#directorygroup)

### Tenant.Realm.Directory.Credential

Authentication factor for identity verification. Supports multiple
credential types (password, TOTP, FIDO, biometric, certificate) with
type-specific data and validation rules.

**Allowed Operations:**
- `create` - Issue new credential
- `read` - Retrieve credential by ID
- `list` - Query credentials (by identity_id)
- `delete` - Revoke and remove credential (immutable, cannot update)

```
Credential
├── id: UUID                     /// system-level identifier
└── data: Variant                /// credential type and data
    | PASSWORD: PasswordCredential
    | TOTP: TOTPCredential
    | EMAIL: EmailCredential
    | SMS: SMSCredential
    | FIDO: FIDOCredential
    | BYID: BYIDCredential
    | X509: X509Credential
    | SSH: SSHCredential
    | JWT: JWTCredential
    | PGP: PGPCredential
```

[Schema](schemas/directory.md#directorycredential)

### Tenant.Realm.IdP

Identity provider for authentication, authorization, and audit over
identities from linked directories for integrated applications and
clients.

**Allowed Operations:**
- `create` - Provision new IdP
- `read` - Retrieve IdP by ID
- `list` - Query IdPs within realm
- `update` - Modify IdP configuration or linked directories
- `delete` - Remove IdP and all associated sessions/grants

```
IdP
├── id: UUID                     /// system-level identifier
├── name: String                 /// unique within realm
└── directories: List[UUID]      /// references to directories
```

[Schema](schemas/idp.md#idp)

### Tenant.Realm.IdP.Session

Active IdP session entity tracking authenticated user-agent sessions.
Represents the authenticated session between user and IdP.

**Allowed Operations:**
- `read` - Retrieve session by ID
- `list` - Query active sessions (created via authentication flow)

```
Session
├── id: UUID                     /// system-level identifier
├── identity_id: UUID            /// reference to identity
├── auth_method: Enum            /// PASSWORD, MFA, PASSKEY, WEBAUTHN,
│                                /// SSO, FEDERATED
├── binding_method: Enum         /// COOKIE, BEARER, DPOP
├── issued_at: Datetime          /// when session was created
├── last_activity: Datetime      /// last session use (idle timeout)
├── expires_at: Datetime         /// when session expires
├── source_ip: String            /// source IP address
└── user_agent: String           /// browser/device info
```

[Schema](schemas/idp.md#idpsession)

### Tenant.Realm.IdP.Authorization

User authorization/consent for a client to access resources. Persistent
record of user consent with specific scopes. Foundation for issuing
tokens. Deletion moves record to retirement table for audit.

**Allowed Operations:**
- `create` - Grant authorization (user consent via OAuth flow)
- `read` - Retrieve authorization by ID
- `list` - Query active authorizations
- `delete` - Revoke authorization (remove user consent)

```
Authorization
├── id: UUID                     /// system-level identifier
├── client_id: UUID              /// reference to client
├── identity_id: UUID            /// reference to identity
├── scope: String                /// authorized scopes
├── audience: String             /// optional, resource server restriction
├── consent_method: Enum         /// EXPLICIT, IMPLICIT, ADMIN, FEDERATED
├── granted_at: Datetime         /// when authorization was granted
├── last_used_at: Datetime       /// last token issuance (stale detection)
├── expires_at: Datetime         /// optional, no expiration if absent
└── metadata: JSON               /// optional, extensible tracking
```

[Schema](schemas/idp.md#idpauthorization)

### Tenant.Realm.IdP.Token

Active tokens issued by the IdP across all protocols. Tracks concrete
credentials issued based on authorizations. Active table only - deletion
moves to retirement table.

**Allowed Operations:**
- `read` - Retrieve token by ID
- `list` - Query active tokens (issued via protocol flows)

```
Token
├── id: UUID                     /// system-level identifier
├── authorization_id: UUID       /// reference to authorization
├── client_id: UUID              /// reference to client
├── identity_id: UUID            /// reference to identity
├── parent_id: UUID              /// null: first token from fresh auth
│                                /// non-null: refreshed from parent
│                                /// (parent in retirement table)
├── token_type: Variant          /// token format/protocol
│   | JWT: JWTToken              /// JWT access token (OAuth/OIDC)
│   | SAML: SAMLAssertion        /// SAML assertion
│   | OIDC_ID: OIDCIDToken       /// OIDC ID token
├── grant_type: Enum             /// AUTHORIZATION_CODE, REFRESH_TOKEN,
│                                /// CLIENT_CREDENTIALS, DEVICE_CODE,
│                                /// TOKEN_EXCHANGE
├── audience: String             /// resource server restriction
├── issued_at: Datetime          /// when token was issued
├── expires_at: Datetime         /// when token expires
├── source_ip: String            /// requester IP address
└── user_agent: String           /// requester user agent
```

[Schema](schemas/idp.md#idptoken)

### Tenant.Realm.IdP.Protocol.OAuth2

IdP-level OAuth2 configuration state defining protocol behavior.

**Allowed Operations:**
- `create` - Configure OAuth2 protocol for IdP
- `read` - Retrieve OAuth2 configuration
- `update` - Modify OAuth2 settings
- `delete` - Remove OAuth2 protocol configuration

```
OAuth2
├── Endpoints
│   ├── Authorize
│   │   ├── path: String = "/authorize"
│   │   └── post: Boolean = true
│   └── Token
│       ├── path: String = "/token"
│       └── body_auth: Boolean = true
├── Grants
│   ├── auth_code: Boolean = true
│   └── client_creds: Boolean = false
├── PKCE
│   ├── require_public: Boolean = true
│   ├── require_confidential: Boolean = false
│   └── methods: List[Enum(S256, PLAIN)] = [S256]
├── Auth Code
│   ├── ttl: Duration = 60s
│   └── state_required: Boolean = false
└── Token
    ├── ttl: Duration = 1h
    ├── type: Enum(BEARER, DPOP) = BEARER
    └── Refresh
        ├── max_uses: Integer = 0       /// 0=unlimited
        └── ttl: Duration = null        /// null=unlimited
```

[Schema](schemas/idp.md#idpprotocoloauth2)

### Tenant.Realm.IdP.Client

OAuth2/OIDC client application with protocol-specific configuration.
Represents applications and services that authenticate users through
this IdP.

**Allowed Operations:**
- `create` - Register new client application
- `read` - Retrieve client by ID
- `list` - Query clients within IdP
- `update` - Modify client configuration
- `delete` - Remove client and revoke all tokens

```
Client
├── id: UUID                     /// system-level identifier
├── name: String                 /// for admin reference
└── description: String          /// purpose/notes for admin
```

[Schema](schemas/idp.md#idpclient)

### Tenant.Realm.IdP.Client.Context

Administrator-defined evaluation context for client-specific custom
scopes. Defines the structure of ctx variables available when evaluating
client-specific scope constraints. Enables error checking to ensure
administrators build scopes they can verify.

**Allowed Operations:**
- `create` - Define initial context for client
- `read` - Retrieve context definition
- `update` - Modify context structure
- `delete` - Remove context definition

```
Context
└── attributes: List[ContextAttribute]
    ├── name: String             /// attribute name (e.g., department)
    ├── type: Type               /// attribute type
    └── description: String      /// attribute purpose
```

[Schema](schemas/idp.md#idpclientcontext)

### Tenant.Realm.IdP.Client.Scopes

Authorization scope defining permissions granted to tokens issued for
this client. Similar to Tenant.Realm.API.Scope but client-specific,
evaluated using the client's custom Context.

**Allowed Operations:**
- `create` - Define new client-specific scope
- `read` - Retrieve scope by ID
- `list` - Query scopes for this client
- `update` - Modify scope permissions
- `delete` - Remove scope

```
Scopes
├── id: UUID                     /// system-level identifier
├── name: String                 /// scope identifier (e.g., read:orders)
├── description: String          /// optional human-readable description
└── permissions: List[Permission] /// list of permissions
                                 /// (uses client-specific context)
```

[Schema](schemas/idp.md#idpclientscopes)

### Tenant.Realm.IdP.Client.Protocol.OAuth2

Client-specific OAuth2 configurations and overrides for application
integration.

**Allowed Operations:**
- `create` - Configure OAuth2 protocol for client
- `read` - Retrieve client OAuth2 configuration
- `update` - Modify client-specific OAuth2 settings
- `delete` - Remove client OAuth2 configuration

```
OAuth2
├── id: String                   /// system-level identifier
├── name: String
├── type: Enum(PUBLIC, CONFIDENTIAL)
├── auth: Enum(CLIENT_SECRET, JWT)
├── redirect_uris: List[URI]
├── scope: String
└── Overrides                    /// optional
    ├── Grants
    │   ├── auth_code: Boolean
    │   └── client_creds: Boolean
    ├── PKCE
    │   ├── require_public: Boolean
    │   ├── require_confidential: Boolean
    │   └── methods: List[Enum(S256, PLAIN)]
    ├── Auth Code
    │   ├── ttl: Duration
    │   └── state_required: Boolean
    └── Token
        ├── ttl: Duration
        ├── type: Enum(BEARER, DPOP)
        └── Refresh
            ├── max_uses: Integer        /// 0=unlimited
            └── ttl: Duration            /// null=unlimited
```

[Schema](schemas/idp.md#idpclientprotocoloauth2)

### Tenant.Realm.IdP.ExtIdP

Third-party IdP for federation. Delegates authentication and
authorization to on-premise providers (Microsoft ADFS), legacy
providers (Okta), or specialty identity assurance providers (Clear,
NameTag).

**Allowed Operations:**
- `create` - Configure external IdP integration
- `read` - Retrieve external IdP configuration by ID
- `list` - Query configured external IdPs
- `update` - Modify external IdP settings
- `delete` - Remove external IdP integration

```
ExtIdP
├── id: UUID                     /// system-level identifier
└── name: String                 /// for admin reference
```

[Schema](schemas/idp.md#idpextidp)

### Tenant.Realm.Device

Physical or virtual device with credentials spanning directories, realms,
or tenants. Supports multi-credential scenarios like consultant devices
with access to multiple organizations.

**Allowed Operations:**
- `read` - Retrieve device by ID
- `list` - Query devices (created via device enrollment protocols)

```
Device
├── id: UUID                     /// system-level identifier
├── name: String                 /// for admin reference
├── architecture: String         /// e.g., x86_64, arm64, aarch64
├── abi: String                  /// e.g., gnu, musl, msvc
├── os: String                   /// e.g., linux, darwin, windows
└── credentials: List[UUID]      /// references to credentials
                                 /// (may span tenants/realms/dirs)
```

[Schema](schemas/device.md)

### Tenant.Realm.Workloads

[TBD] - Workload entity for managing application workloads and services.

### Tenant.Realm.Hooks

[TBD] - Hook entity for extensibility and custom event handling.

### Tenant.Realm.Providers.Log/SIEM

Export integration to log or SIEM provider. All events passing filters
sent to provider.

**Allowed Operations:**
- `create` - Configure new log/SIEM integration
- `read` - Retrieve provider configuration by ID
- `list` - Query configured providers
- `update` - Modify provider configuration or filters
- `delete` - Remove log/SIEM integration

```
Log/SIEM
├── id: UUID                           /// system-level identifier
├── name: String                       /// for admin reference
└── filters: List[Expression[Boolean]] /// events matching filters
                                       /// are exported
```

[Schema](schemas/providers.md#providerslogsiem)

