# Local secrets and release hygiene

Store real credentials in `.secrets/bfcl.env` or an explicitly selected external file. Keep permissions private. `.gitignore` is a convenience, not an access-control system: `git add -f` can bypass it and it cannot erase tracked history.

Keep local datasets, databases, models, downloaded benchmarks and generated outputs out of version control. Never paste private keys, provider tokens or private conversations into bug reports, AI prompts or examples.

Before publication, scan both tracked files and all Git history with Gitleaks. Review ambiguous matches semantically. Test fixtures containing token-shaped strings are synthetic and constructed in code. Local model clients may use the literal `EMPTY` or `local` for servers without authentication; those literals are not external provider credentials and must never be used to expose a server to the internet.

If a real secret was exposed, revoke or rotate it at its provider. Removing the latest file does not remove it from earlier commits. A clean new repository must not import branches, tags or objects from the private archive.
