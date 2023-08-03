# PostBOUND documentation

The documentation is based on Sphinx, so make sure to have this package installed.

In order to generate the documentation files, run the following commands:

```sh
$ sphinx-apidoc --force \
                --ext-autodoc \
                --maxdepth 4 \
                --module-first \
                -o source/generated \
                ../postbound
$ make html
```
