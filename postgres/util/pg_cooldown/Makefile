# contrib/pg_cooldown/Makefile

MODULE_big = pg_cooldown
OBJS = \
	$(WIN32RES) \
	pg_cooldown.o

EXTENSION = pg_cooldown
DATA = pg_cooldown--0.1.sql
PGFILEDESC = "pg_cooldown - manually remove relation data from the system buffer cache"

ifdef USE_PGXS
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
else
subdir = contrib/pg_cooldown
top_builddir = ../..
include $(top_builddir)/src/Makefile.global
include $(top_srcdir)/contrib/contrib-global.mk
endif
