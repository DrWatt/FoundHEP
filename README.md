# FoundHEP

**Foundational models for High-Energy Physics data compression**

FoundHEP is an experimental Python package for exploring learned representations and compression methods for High-Energy Physics (HEP) data.

The project currently combines **Transformer-based sequence models**, **vector quantization**, and **variational autoencoder components** using Keras and TensorFlow. The aim is to provide building blocks for studying compact learned representations of HEP datasets.

> [!WARNING]
> **FoundHEP is currently an alpha-stage research project.**
>
> APIs may change without notice, and some components—particularly the end-to-end VQ-VAE implementation—are still under development.


## Architecture

FoundHEP currently contains three main groups of components.

### Transformer encoder

`TransEncoder` implements a Transformer-style encoder block:

```text
Input
  │
  ├─ LayerNorm
  │
  ├─ Multi-Head Self-Attention
  │    └─ Learned per-head scaling
  │
  ├─ LayerNorm + Dropout
  │
  ├─ Residual connection
  │
  ├─ LayerNorm
  │
  ├─ Dense + SiLU
  │
  ├─ LayerNorm + Dense + Dropout
  │
  └─ Residual connection
  │
Output
```

### Transformer decoder

`TransDecoder` extends this structure with:

```text
Input
  │
  ├─ Causal Self-Attention
  │
  ├─ Residual connection
  │
  ├─ Cross-Attention ───── Encoder representation
  │
  ├─ Residual connection
  │
  ├─ Feed-Forward Network
  │
  └─ Residual connection
  │
Output
```

### Head-scaled multi-head attention

`HeadScaledMultiHeadAttention` extends Keras' multi-head attention mechanism with one trainable scalar per attention head.

Each scalar is initialized to `1`, making the initial operation equivalent to ordinary multi-head attention while allowing training to learn the relative contribution of individual heads.

Conceptually:

$$
\begin{aligned}
\mathrm{head}_1 &\times \alpha_1 \\
\mathrm{head}_2 &\times \alpha_2 \\
&\vdots \\
\mathrm{head}_n &\times \alpha_n
\end{aligned}
$$

where each $\alpha$ is a learned parameter.

## Vector quantization

FoundHEP also contains experimental vector-quantization components inspired by VQ-VAE approaches.

The current implementation includes:

* `VectorQuantizer`
* `VectorQuantizerEMA`
* latent sampling utilities
* dense encoder/decoder components
* an experimental `VQVAE` model

Vector quantization maps continuous latent representations onto entries in a learned discrete codebook, making it a natural mechanism for exploring learned compression.



## Repository structure

```text
FoundHEP/
├── LICENSE
├── pyproject.toml
└── src/
    └── FoundHEP/
        ├── __init__.py
        ├── TransDer.py
        ├── VQVAE.py
        ├── vquant.py
        └── custom_layers.py
```

### `TransDer.py`

Contains the main Transformer components:

* `TransEncoder`
* `TransDecoder`

### `custom_layers.py`

Contains custom neural-network layers, including:

* `HeadScaledMultiHeadAttention`

### `VQVAE.py`

Contains experimental compression and latent-representation components, including:

* `VectorQuantizer`
* `VectorQuantizerEMA`
* `Sampling`
* `Encoder`
* `Decoder`
* `VQVAE`

## Development

Install the project in editable mode:

```bash
python -m pip install -e .
```

Optional documentation dependencies can be installed with:

```bash
python -m pip install -e ".[docs]"
```

The documentation dependency set includes Sphinx, NumPyDoc, Sphinx Design, and the PyData Sphinx Theme.

## Contributing

Contributions, bug reports, feature proposals, and discussions are welcome.

For larger changes, consider opening an issue first to discuss the proposed design and its relationship to the project's HEP compression goals.

## License

FoundHEP is distributed under the **Apache License 2.0**.

See `LICENSE` for the full license text.

## Acknowledgements

Parts of the vector-quantization implementation are based on ideas from the original **VQ-VAE** work and related Keras and DeepMind/Sonnet implementations.

## Citation

FoundHEP is currently an early-stage research project and does not yet provide a formal citation.

If you use FoundHEP in research, please cite the repository until a publication or citation file is provided.

