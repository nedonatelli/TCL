Special Functions
=================

This example demonstrates special mathematical functions including Bessel functions and their applications.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/special_functions.html"></iframe>
   </div>

Overview
--------

Special functions are fundamental to many engineering applications:

- **Signal processing**: Filter design and analysis
- **Electromagnetics**: Waveguide mode calculations
- **Acoustics**: Circular membrane vibrations
- **Optics**: Diffraction patterns

Bessel Functions
----------------

**First Kind (J_n)**
   - Solutions to Bessel's differential equation
   - Finite at the origin
   - Oscillatory behavior for positive arguments

**Second Kind (Y_n)**
   - Also called Neumann functions
   - Singular at the origin
   - Independent solution to Bessel's equation

**Key Properties**
   - J_0(0) = 1, J_n(0) = 0 for n > 0
   - Recurrence relations connect different orders
   - Zeros are important for boundary value problems

Applications
------------

**Circular Drum Vibrations**
   - Bessel function zeros determine mode frequencies
   - J_0 zeros: fundamental modes
   - Higher orders: more complex patterns

**Cylindrical Waveguides**
   - TE and TM mode cutoff frequencies
   - Field patterns in circular cross-section
   - Microwave and optical applications

**Bessel Filters**
   - Maximally flat group delay
   - Linear phase response
   - Named after Bessel functions

**Boundary Value Problems**
   - Heat conduction in cylinders
   - Electromagnetic fields
   - Quantum mechanics (spherical wells)

Bessel Zeros
------------

The zeros of Bessel functions are critical values:

- J_0 zeros: 2.405, 5.520, 8.654, 11.792, ...
- Used in filter design and mode analysis
- Computed with ``bessel_zeros()``

Code Highlights
---------------

The example demonstrates:

- Bessel function evaluation with ``besselj()`` and ``bessely()``
- Multiple orders (J_0 through J_4)
- Zero finding with ``bessel_zeros()``
- Visualization of function behavior

Source Code
-----------

.. literalinclude:: ../../../examples/special_functions_demo.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/special_functions_demo.py

See Also
--------

- :doc:`signal_processing` - Signal processing applications
- :doc:`transforms` - Mathematical transforms
