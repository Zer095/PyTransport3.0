PyTransport C++ Library Reference
=================================

This section provides detailed reference documentation for the core C++ components
of the PyTransport library. This documentation is generated directly from the
Doxygen comments in the C++ source code.

PyTrans.cpp
-----------

This file contains the main implementations of the C++ functions that are
exposed to the Python layer.

.. doxygenfile:: PyTrans.cpp


Header Files
------------

Here are some key header files and the entities they define.

model.h
^^^^^^^

The `model.h` header defines the base interface for inflationary models.

.. doxygenfile:: cppsrc/model.h


evolve.h
^^^^^^^^

The `evolve.h` header contains declarations for functions related to the
evolution of various cosmological quantities.

.. doxygenfile:: cppsrc/evolve.h

moments.h
^^^^^^^^^

The `moments.h` header defines structures and functions related to
moment calculations.

.. doxygenfile:: cppsrc/moments.h

fieldmetric.h
^^^^^^^^^^^^^

The `fieldmetric.h` header deals with the field-space metric calculations.

.. doxygenfile:: cppsrc/fieldmetric.h

potential.h
^^^^^^^^^^^

The `potential.h` header defines the potential functions and their derivatives.

.. doxygenfile:: cppsrc/potential.h



