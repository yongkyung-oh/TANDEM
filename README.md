# TANDEM: Temporal Attention-guided Neural Differential Equations for Missingness in Time Series Classification

Handling missing data in time series classification remains a significant challenge across domains such as healthcare and sensor analytics. Traditional approaches often depend on imputation, which may introduce bias or fail to capture the underlying temporal dynamics.  

**TANDEM** (*Temporal Attention-guided Neural Differential Equations for Missingness*) is a framework that integrates raw observations, interpolated control paths, and continuous latent dynamics through a novel temporal attention mechanism. By doing so, the model focuses on the most informative aspects of the data and achieves robust classification performance under missingness.  

Our experiments on 30 benchmark datasets and a real-world medical dataset demonstrate that TANDEM outperforms existing state-of-the-art methods, while also offering interpretability in how missing values are handled.

---

## **Code architecture**

The repository is organized into two main components:

- `torch-ists`: Independent library for irregular time series, motivated by prior works such as Stable Neural SDEs [1] and DualDynamics [2]. 
- `physionet-sepsis`: Experiments on the PhysioNet Sepsis dataset for evaluating TANDEM in a clinical classification task under missingness. Original pipeline was suggested by [3]

> [1] Oh, Y., Lim, D., & Kim, S. (2024). Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data. The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. https://openreview.net/forum?id=4VIgNuQ1pY

> [2] Oh, Y., Lim, D.-Y., & Kim, S. (2025). DualDynamics: Synergizing Implicit and Explicit Methods for Robust Irregular Time Series Analysis. In T. Walsh, J. Shah, & Z. Kolter (Eds.), AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25—March 4, 2025, Philadelphia, PA, USA (pp. 19730–19739). AAAI Press. https://doi.org/10.1609/AAAI.V39I18.34173

> [3] Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020). Neural controlled differential equations for irregular time series. Advances in Neural Information Processing Systems, 33, 6696-6707.

---

## **Core implementation details**

TANDEM builds upon the foundations of Neural Differential Equations, with several key extensions:

- **Temporal Attention Mechanism**: guides the model to focus on informative segments of irregular and incomplete data.  
- **Neural Differential Equations Backbone**: models continuous latent dynamics, offering flexibility in handling irregular sampling.  
- **Integration of Paths**: combines raw observations and interpolated control paths to mitigate bias from direct imputation.  

---

## **Citation**

If you use this repository, please cite:

```bibtex
@inproceedings{oh_tandem_2025,
  title        = {{TANDEM}: Temporal Attention-guided Neural Differential Equations for Missingness in Time Series Classification},
  author       = {Oh, YongKyung and Lim, Dong-Young and Kim, Sungil and Bui, Alex A. T.},
  year         = {2025},
  booktitle    = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM 2025)},
  address      = {Seoul, Republic of Korea},
  month        = {November},
  note         = {To appear},
}
