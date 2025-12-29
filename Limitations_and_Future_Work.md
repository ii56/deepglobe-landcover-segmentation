## Limitations

Although the proposed land-cover segmentation system demonstrates effective performance on images similar to the DeepGlobe training dataset, several limitations are observed.

First, the model is sensitive to domain shift. It was trained on high-resolution optical satellite imagery with specific color distributions, spatial resolutions, and land-cover characteristics. When applied to out-of-distribution imagery, such as dense forests, mountainous terrain, or images captured under different illumination conditions, segmentation quality degrades noticeably. This behavior is expected in deep learning models and highlights the dependency of performance on training data characteristics.

Second, the system performs single-date semantic segmentation and does not incorporate temporal information. As a result, it cannot reliably perform change detection tasks such as flood mapping or land-use evolution without additional temporal supervision. Any inferred changes between two independently processed images should therefore be interpreted cautiously.

Third, the model relies solely on RGB imagery. Important information present in multispectral or SAR data (e.g., near-infrared bands for vegetation health or radar data for flood detection) is not utilized, limiting robustness in complex environmental conditions.

Finally, class imbalance within the training dataset affects segmentation quality for minority classes, leading to fragmented predictions in certain regions, particularly where land-cover boundaries are subtle or ambiguous.

---

## Future Work

Several extensions can be explored to improve the system.

Future work could include domain adaptation or fine-tuning using region-specific satellite datasets to improve generalization across diverse terrains and ecosystems. Incorporating data augmentation and color normalization techniques would further reduce sensitivity to illumination and sensor variations.

To enable reliable flood detection or land-cover change analysis, the system can be extended to a multi-temporal learning framework, where before-and-after images are jointly processed using change-aware architectures such as Siamese networks.

Additionally, integrating multispectral or SAR data would significantly enhance performance in vegetation-dense and cloud-covered regions. This would allow the model to capture physical properties of land surfaces that are not visible in RGB imagery alone.

Finally, future versions of the system could include confidence estimation and uncertainty visualization, enabling users to better understand prediction reliability in real-world applications.