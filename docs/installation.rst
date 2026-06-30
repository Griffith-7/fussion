Installation
============

Requirements
------------

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+

Install from GitHub
-------------------

.. code-block:: bash

    pip install git+https://github.com/Griffith-7/fussion.git

Editable install (for development)
-----------------------------------

.. code-block:: bash

    git clone https://github.com/Griffith-7/fussion.git
    cd fussion
    pip install -e ".[dev]"

Optional Dependencies
---------------------

For CLIP encoder:

.. code-block:: bash

    pip install ftfy regex

For Whisper encoder:

.. code-block:: bash

    pip install openai-whisper

For video encoder:

.. code-block:: bash

    pip install av
