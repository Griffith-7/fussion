Usage
=====

Quick Start
-----------

.. code-block:: python

    from fussion import CrossModalMerger, get_encoder, list_modalities

    print(list_modalities())  # ['image', 'video', 'code', 'text']

    encoder = get_encoder("clip")
    merger = CrossModalMerger(
        encoder=encoder,
        llm_name="distilgpt2",
        encoder_dim=512,
        llm_dim=768,
    )

    # Generate text conditioned on an image
    text = merger.generate("path/to/image.jpg", prompt="Describe this image:")
    print(text)

Internal Fusion (Flamingo-style)
--------------------------------

.. code-block:: python

    from fussion import FusionLLM, train_fusion

    model = FusionLLM(encoder_dim=512, llm_name="distilgpt2", every_k_layers=4)
    train_fusion(model, encoder, train_src, train_tgt, val_src, val_tgt)
    text = model.generate(encoder, "path/to/image.jpg", prompt="Describe:")
    print(text)

Datasets
--------

.. code-block:: python

    from fussion.datasets import make_shape_dataset, make_text_dataset

    shape_data = make_shape_dataset(n=50)
    text_data = make_text_dataset(n=50)

CLI
---

.. code-block:: bash

    python -m fussion train --encoder clip --llm distilgpt2 --epochs 5
    python -m fussion generate --checkpoint checkpoints/model.pt --input image.jpg

List modalities:

.. code-block:: bash

    python -m fussion list-modalities
