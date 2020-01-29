# libartificial

This is a small pure header-only and zero-dependency C library for arbitrarily deep neural networks used by Fuzznets P.C.

It is CPU only at the moment, with plans to incorporate Metal, CUDA and openCL functionality.

Adam, AdaGrad etc. will be added soon. There are plans to extend it for CNNs, LSTMs and Graph Networks.

## Prerequisites

Works with gcc and clang.

All functions and their examples should run through lldb, gdb and valgrind without any errors.

The code follows the Visual Studio C styling guide.

## Examples

- MLP regression:

```
make test1
```

- MLP classification:

```
make test2
```

If you want to compile them all then just do

```
make
```

## License

Copyright (c) 2020 [Fuzznets P.C](https://www.fuzznets.com) and [Dim Karoukis](https://www.dkaroukis.com). All rights reserved. The software is distributed under the terms of the [GNU General Public Licence version 3](https://www.gnu.org/licenses/gpl-3.0.html). For more info see the [LICENCE](/LICENCE) file.
