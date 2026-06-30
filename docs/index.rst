Welcome to fussion's documentation!
=====================================

**fussion** connects any encoder to any frozen LLM — cross-modal fusion with zero-init bridges.

.. code-block:: python

    from fussion import CrossModalMerger, list_modalities, get_encoder

    print(list_modalities())  # ['image', 'video', 'code', 'text']
    encoder = get_encoder("clip")
    merger = CrossModalMerger(encoder=encoder, llm_name="distilgpt2")
    text = merger.generate("cat.jpg")
    print(text)


Features
--------

* **Multi-modal fusion** — Connect vision, video, code, or text encoders to any LLM.
* **Zero-init bridge** — Starts as identity, fine-tuned via next-token prediction.
* **Internal fusion (Flamingo-style)** — Cross-attention layers inserted every K transformer layers.
* **External bridge** — Standalone ``CrossModalMerger`` for lightweight fine-tuning.
* **Pre-built encoders** — CLIP (image), Whisper (audio), Video, CodeBERT (code), Text LLM.
* **Synthetic datasets** — Built-in ``make_shape_dataset``, ``make_video_dataset``, ``make_code_dataset``, ``make_text_dataset``.
* **CLI** — ``python -m fussion train``, ``python -m fussion generate``.

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   usage
   api
