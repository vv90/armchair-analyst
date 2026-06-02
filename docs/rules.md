# Coding Rules

These rules apply to production code. Tests may deliberately exercise panic paths, invalid states, and failure behavior.

## Panic Discipline

- Absolutely no code outside tests should ever panic.
- Any function, whether built in, locally defined, or provided by a crate or library, must be understood before use with respect to panic behavior.
- If a function can panic, that possibility must be explicitly and exhaustively handled.
- Prefer non-panicking functions and APIs whenever they are available.
- Do not use panic-prone convenience APIs in production code when a checked alternative exists.
- Do not rely on assumptions such as "this cannot happen" unless the type system or prior exhaustive validation makes the invalid state unrepresentable.

## Pure Logic

- Any non-trivial logic should be pure.
- Non-trivial pure logic should be covered by property tests.
- Pure logic should be deterministic: the same inputs must always produce the same outputs.
- Pure logic should not perform I/O, access clocks, generate randomness, mutate external state, spawn work, or depend on environment state.

## Impure Logic

- Any impure logic should be trivial.
- If impure logic becomes non-trivial, separate it into a pure part and a trivial impure boundary.
- The impure boundary should only translate between the external world and pure domain functions.
- Keep effect execution implementations as small and direct as possible.

## Functional Style

- Always use a functional programming style.
- Avoid mutable values, mutable references, loops, in-place data manipulation, and other imperative-style code.
- Prefer expressions, immutable data, iterator pipelines, total functions, explicit return values, and composition.
- Prefer APIs and data structures that make invalid states unrepresentable.

## Imperative Code Red Flags

If imperative code appears unavoidable, treat it as a strong code smell.

Before proceeding with imperative code:

- Re-evaluate the assumptions that made it seem unavoidable.
- Consider whether the design, data model, API boundary, or ownership structure can be changed.
- Investigate whether the imperative portion can be isolated behind a tiny impure boundary.
- Make the concern explicit in code review and documentation.

Do not proceed with non-trivial imperative production code until these assumptions have been deeply investigated.
