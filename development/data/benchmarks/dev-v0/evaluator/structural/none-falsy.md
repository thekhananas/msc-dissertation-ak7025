# Structural difference: missing versus falsy

The evidence task handles optional integers and produces display text. The held-out task handles optional strings and selects between a supplied label and a default. Types, function signatures, outputs, and tests differ while both require an identity check against `None` rather than a broad truthiness check.
