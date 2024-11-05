/*
 * drop-caches.c
 *
 * A utility to run drop_caches without super-user privileges
 * See https://stackoverflow.com/q/13646925/5161760 for details.
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <sys/types.h>

extern void sync(void);

int main(void)
{
  if (geteuid() != 0) {
    fprintf(stderr, "drop-caches: Not root\n");
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "Flushing page cache, dentries and inodes...\n");

  /* First, the traditional three sync calls. Perhaps not needed? */
  sync();
  sync();
  sync();

  /* Now, actually drop the cache */
  FILE* fDropCaches;
  fDropCaches = fopen("/proc/sys/vm/drop_caches", "w");
  if (fDropCaches == NULL) {
    fprintf(stderr, "drop-caches: Could not open /proc/sys/vm/drop_caches\n");
    exit(EXIT_FAILURE);
  }

  if (fprintf(fDropCaches, "3\n") != 2) {
    fprintf(stderr, "drop-caches: Could not write to /proc/sys/vm/drop_caches\n");
    exit(EXIT_FAILURE);
  }

  fclose(fDropCaches);
  fprintf(stderr, "Done flushing\n");
  return 0;
}
