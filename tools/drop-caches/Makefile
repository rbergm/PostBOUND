
drop-caches: drop-caches.c
	cc -o $@ $^
	sudo chown root $@
	sudo chmod u+s $@
	mkdir -p ../bin/
	ln -rsf $@ ../bin/$@

clean:
	rm drop-caches

.PHONY: drop-caches
