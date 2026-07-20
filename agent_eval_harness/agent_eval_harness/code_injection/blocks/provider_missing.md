### 🔧 Provider Listing Not Auto-Detected

The list of configured LLM providers/models was not harvested. Resolve it at runtime.

**What you actually need — four things, nothing else:**

1. Which providers exist — id / display name / kind
2. The model id to use
3. The `base_url` — where the running target answers
4. **How** the target supplies its credential at call time: the mechanism only (e.g. "the app
   injects it server-side", "it reads env var X") — never the value

You never need the key itself: the target authenticates its own outbound calls, and the
dispatch modules address the target, not the provider.

**How to get them:**

1. Prefer the target's own provider/config **service** (see each agent card's constructor
   dependencies for one tagged `llm_provider`) and its masked listing method, or a masked
   HTTP listing endpoint. These strip secrets for you.
2. If you must read the store directly:
   a. Read the **schema first** — column names only, no rows — to find which columns hold
      items 1-3.
   b. **Ask the human for permission**, showing the exact query and the exact column list.
      Wait for approval.
   c. `SELECT` those columns **by name**. Never `SELECT *`, and never a free-form blob column
      (`extra`, `config`, `settings`, `metadata`, `env`): credentials are routinely stored
      *inside* such a column's JSON, so it leaks a key even though the name looks harmless.
3. Show the masked list to the human and ask which provider/model to use (task T004).
4. If a key value appears in your output anyway: stop, report it, and tell the human to rotate
   that credential immediately. Never echo the value again.
