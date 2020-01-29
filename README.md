# libartificial

This is a small pure header-only and zero-dependency C library for arbitrarily deep feedforward neural networks used by Fuzznets P.C.

It is CPU only at the moment, with plans to incorporate CUDA, openCL and Metal functionality.

Adam, AdaGrad etc. will be added soon. There are plans to extend it for CNNs, LSTMs and Graph Networks.

## Prerequisites

Works with gcc and clang on Linux and macOS.

All functions and their examples should run through lldb, gdb and valgrind without any errors.

The code follows the Visual Studio C styling guide.

## Examples

- MLP regression:

```
make test1
./examples/test1
```

- MLP classification:

```
make test2
./examples/test2
```

If you want to compile them all then just run

```
make
```

## License

Copyright (c) 2020 [Fuzznets P.C](https://www.fuzznets.com) and [Dim Karoukis](https://www.dkaroukis.com). All rights reserved.
The software is distributed under the terms described in the [LICENCE](LICENCE) file.
