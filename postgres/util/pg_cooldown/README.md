# pg_cooldown

Postgres extension to remove certain relations from the shared buffer.

## Usage

Install like a normal contrib extension (i.e. using `make && make install` in the extension directory), activate with
`CREATE EXTENSION pg_cooldown;`.

```sql
SELECT pg_cooldown('title');  -- removes relation title
SELECT pg_cooldown('title_pkey');  -- removes primary key index structures for title relation
```
