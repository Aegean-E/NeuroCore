# Decisions

Why this project is built the way it is, oldest first. One entry per decision that changes a
contract, a dependency, a data format or the architecture; IDs are consecutive and never reused.

<!-- Entry format (use the next free ADR-NNN id):
### ADR-NNN: <decision in one line>
- **Date**: YYYY-MM-DD
- **Status**: Accepted
- **Context**: <what forced the decision>
- **Decision**: <what was decided>
- **Consequences**: <what follows, including what it costs>
-->

### ADR-001: Bind the control plane to loopback, opt in to remote/reload, and guard the Host header
- **Date**: 2026-10-20
- **Status**: Accepted
- **Context**: `main.py` started uvicorn on `0.0.0.0` with `reload=True`, so a default launch was
  reachable from any network and restarted on file changes; the Docker publish (`8000:8000`) was
  likewise open to every host interface. The control plane has no authentication, so this exposes
  RCE-capable routes (module hot-load, tool import) to the network.
- **Decision**: `core/security.py` owns the bind policy. The server binds `127.0.0.1` unless
  `NEUROCORE_HOST` opts in; auto-reload is off unless `NEUROCORE_RELOAD` opts in; and a
  Starlette `TrustedHostMiddleware` accepts only loopback hosts plus `testserver` unless
  `NEUROCORE_ALLOWED_HOSTS` opts in. `docker-compose.yml` publishes `127.0.0.1:8000:8000` while
  the container binds `0.0.0.0` internally.
- **Consequences**: A local launch keeps working unchanged. Reaching the app from another machine
  now needs two deliberate acts (a bind opt-in and, when applicable, an allowed-host entry).
  Docker is only reachable through the host loopback address until the publish mapping is edited.

### ADR-002: Extract marketplace archives through a path-confined helper
- **Date**: 2026-10-20
- **Status**: Accepted
- **Context**: `import_marketplace_item_to_neurocore` called `zipfile.extractall` (and a manual
  member loop) on attacker-controlled packages, so a member named `../evil.py`, an absolute path,
  or a symlink could write anywhere the process can reach (zip-slip).
- **Decision**: All module extraction goes through `core/archive.extract_zip_safely`, which rejects
  absolute members, `..` traversal, drive-qualified names, symlink entries, and any resolved path
  outside the destination. An unsafe package returns HTTP 400 and writes nothing.
- **Consequences**: Legitimate single-root packages are still unwrapped into `modules/<name>/`. A
  package that relies on writing outside its own folder is refused, which is the intended
  contract. Read-only member access (file preview) is unchanged because it never writes.

### ADR-003: Allowlist first-party sandbox imports and block dunder traversal
- **Date**: 2026-10-20
- **Status**: Accepted
- **Context**: The tool sandbox passed through every `modules.*` import, so tool code could
  `import modules.tools.sandbox` and read that module's real `os`/`sys` globals; dunder traversal
  (`().__class__.__mro__[1].__subclasses__()`) could also reach the real interpreter.
- **Decision**: Passthrough is an explicit allowlist (`modules.calendar.events`) rather than a
  prefix, a bare `modules` import is refused, and the AST gate rejects the dunder attributes,
  subscript keys and `getattr`/`setattr` names that reach the interpreter object graph.
- **Consequences**: Tools still import the one first-party module they need; the sandbox itself is
  no longer importable. Determined code with computed attribute strings is still bounded by the
  subprocess isolation and static gate, which is best-effort in-process Python.

### ADR-004: Keep secrets out of config exports and debug logs
- **Date**: 2026-10-20
- **Status**: Accepted
- **Context**: `/settings/export/config` returned the whole `settings.json` (including
  `llm_api_key`), and `DebugLogger.log` stored and printed arbitrary `details` dicts verbatim.
- **Decision**: `core/security` gains `redact_secrets` (mask, for logs) and `strip_secrets`
  (drop, for exports). The debug logger redacts at its boundary; the config export strips secrets
  by default and only includes them when `NEUROCORE_EXPORT_SECRETS` opts in.
- **Consequences**: Exports no longer round-trip API keys, so a re-import cannot overwrite a real
  key with a placeholder. Full backups need the explicit opt-in. Sensitivity is by known name or
  `*_key`/`*_token`/`*_password`/`*_secret` suffix so ordinary fields are untouched.

### ADR-005: Escape stored content in server-rendered HTML and innerHTML sinks
- **Date**: 2026-10-20
- **Status**: Accepted
- **Context**: Two module routers built HTML by string interpolation
  (`modules/memory/router.py`, `modules/memory_browser/router.py`), and the marketplace templates
  placed stored other-user text (comment bodies, uploader names, notifications) into `innerHTML`.
- **Decision**: The hand-built HTML interpolations call `html.escape`, and the marketplace
  templates define and use `escapeHtml` for every stored field they insert into `innerHTML`. The
  Jinja environment's autoescape is asserted by a regression test.
- **Consequences**: Stored comment/notification text renders as text, not markup. Field-by-field
  escaping must be maintained when new dynamic sinks are added; the chat view's `marked.parse`
  markdown rendering was left unchanged and is noted as remaining work.

### ADR-006: Require an opt-in API token for remote clients and reject cross-origin writes
- **Date**: 2026-10-20
- **Status**: Accepted
- **Context**: The control plane had no authentication or CSRF protection, so once reachable it
  accepted any state-changing request, including from a browser it did not originate.
- **Decision**: A `ControlPlaneGuard` middleware rejects state-changing requests whose `Origin`
  header is neither same-origin nor an explicitly trusted origin, and requires an
  `X-API-Token`/`Authorization: Bearer` match (constant-time) for non-loopback clients when
  `NEUROCORE_API_TOKEN` is set.
- **Consequences**: The local workflow and the test client are unchanged (no token configured,
  same-origin or absent `Origin`). A remote bind now needs `NEUROCORE_API_TOKEN`; because the host
  guard already blocks non-loopback hosts by default, both must be opted in deliberately. The
  token is checked per request so it can be rotated by restart, not mutation.