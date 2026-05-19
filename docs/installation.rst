.. _installation:

Installation
============

Requirements
------------

PROFE requires **Python ≥ 3.11, < 3.12** and the following dependencies
(managed automatically via ``pip``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Version constraint
   * - astropy
     - ``== 5.3.2``
   * - scipy
     - ``>= 1.10, < 2.0``
   * - matplotlib
     - ``>= 3.10.3, < 4.0.0``
   * - tqdm
     - ``>= 4.67.1, < 5.0.0``
   * - pandas
     - ``>= 2.3.1, < 3.0.0``
   * - mc3
     - ``>= 3.2.1, < 4.0.0``
   * - photutils
     - ``>= 2.2.0, < 3.0.0``
   * - numpy
     - ``>= 1.24, < 2.0``
   * - astroquery
     - ``>= 0.4.7``
   * - beautifulsoup4
     - ``>= 4.12.0``
   * - requests
     - ``>= 2.31.0``


Install from PyPI
-----------------

.. code-block:: bash

   pip install profe


Development Installation
------------------------

For a development environment that includes testing, linting, and profiling tools:

.. code-block:: bash

   git clone https://github.com/s-paez/profe.git
   cd profe
   pip install -e ".[dev]"


External Setup
--------------

Some PROFE features require external credentials. These files must be placed at
the **root of your working directory** (where you run ``profe``).

WCS Solving (Astrometry.net)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Automated WCS solving requires an `Astrometry.net <https://nova.astrometry.net/>`_ API key:

1. Create a file named ``.astrometry_key``.
2. Paste your 16-character API key inside the file (no extra whitespace).

.. code-block:: text

   your16charAPIkey

Transit Timing (TTF)
^^^^^^^^^^^^^^^^^^^^

The TESS Transit Finder integration requires Swarthmore TTF credentials:

1. Create a file named ``.ttf_credentials``.
2. Write your credentials in ``username:password`` format.

.. code-block:: text

   myuser:mypassword

ExoFOP Uploads
^^^^^^^^^^^^^^

Automated uploads to ExoFOP require valid account credentials:

1. Create a file named ``.exofop_credentials``.
2. Write your credentials in ``username:password`` format.

.. code-block:: text

   myuser:mypassword

.. note::

   All features degrade gracefully: if the credential files are missing,
   PROFE logs a warning and skips the corresponding step.
