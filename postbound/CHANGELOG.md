# Changelog

Changelog will start once the high-level refactoring of PostBOUND is completed. Notice however, that PostBOUND should
still be considered to be in beta state. Therefore, a number of interfaces will still be changed frequently and changes
will often be breaking changes by definition. If you use PostBOUND in your research, make sure to update carefully.


## Version 0.0.2-beta

- The Postgres interface now tries to be smart about GeQO usage. If a query contains elements that would be overwritten by the
GeQO optimizer, GeQO is disabled for the current query. Afterwards, the original GeQO configurations is restored. At a later
point, this behaviour could be augmented to handle all sorts of side effects and restore the original configuration.
- The Postgres `connect` method now re-uses existing (pooled) instances by default. If this is not desired, the `fresh`
parameter can be set.
