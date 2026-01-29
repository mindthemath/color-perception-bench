Local model was nomic-embed-v1.5

First jina run was not using proper structure with "text" and "image" keys inside of list of dicts:

The first run was sending:
```
{"input": ["text1", "text2", ...]}  // Plain strings
{"input": ["base64img1", "base64img2", ...]}  // Plain base64 strings
```

and the second run (which had a higher max) was sending:
```
{"input": [{"text": "text1"}, {"text": "text2"}, ...]}  // Wrapped in objects
{"input": [{"image": "base64img1"}, {"image": "base64img2"}, ...]}  // Wrapped in objects
```

| timestamp | model_name | n_colors | mean_distance | median_distance | std_distance | variance | min_distance | max_distance |
|-----------|------------|----------|---------------|-----------------|--------------|----------|--------------|--------------|
| 2026-01-15T16:05:00.166303 | nomic-embed-v1.5 | 949 | 0.935270 | 0.932488 | 0.016109 | 0.000259 | 0.899133 | 0.994725 |
| 2026-01-15T17:03:14.620777 | jina-clip-v2 | 949 | 0.718174 | 0.719027 | 0.038234 | 0.001462 | 0.576830 | 0.853418 |
| 2026-01-15T17:23:22.312932 | jina-clip-v2 | 949 | 0.681067 | 0.664699 | 0.056226 | 0.003161 | 0.576200 | 0.922236 |
| 2026-01-15T18:22:44.672770 | jina-embeddings-v4 | 949	| 0.307344 | 0.294318 | 0.063753 | 0.004064 | 0.193064 | 0.527237 |



---
raw data:

timestamp	model_name	n_colors	mean_distance	median_distance	std_distance	variance	min_distance	max_distance
2026-01-15T16:05:00.166303	local-default	949	0.935270	0.932488	0.016109	0.000259	0.899133	0.994725
2026-01-15T17:03:14.620777	jina-clip-v2	949	0.718174	0.719027	0.038234	0.001462	0.576830	0.853418
2026-01-15T17:23:22.312932	jina-clip-v2	949	0.681067	0.664699	0.056226	0.003161	0.576200	0.922236


---

FOLLOWUP

# Foundations

Since Jina v4 uses the Qwen2.5-VL backbone, their most direct rivals are the models coming out of Alibaba's Qwen Team themselves.


- Qwen3-VL (2026 Release): As of early 2026, Qwen3-VL is the primary open-weights challenger. While Jina specializes in the embedding (vector) output, Alibaba's base models are extremely strong at the same "visual thought" alignment. Qwen's advantage is scale; they offer much larger versions that can capture even more minute details of texture and weave. https://openlm.ai/qwen3-vl/

- BGE-VL (BAAI): The Beijing Academy of Artificial Intelligence (creators of BGE-M3) released a visual-text model specifically for retrieval. Like Jina, it focuses on high-precision "Retrieval" rather than just "Chatting about an image."


# The "Fine-Grained" Specialists (Research Leaders)


For your specific interest in color, pattern, and weave, the following models are the "state-of-the-art" in research for 2026:


- MegaCOIN (2025): This is a specialized model and dataset (released Dec 2024/Jan 2025) specifically designed to fix the "color blindness" of models like GPT-4o. It focuses on Medium-Grained Color Perception. If you are building a fabric search engine, the MegaCOIN-instructed models are the primary alternative to Jina for ensuring that "Navy Blue" isn't confused with "Midnight Black." https://arxiv.org/html/2412.03927v1

- FG-CLIP (Fine-Grained CLIP): An evolved version of the original CLIP architecture that uses 1.6 billion "long-caption" pairs. Traditional CLIP models failed at "weave and pattern" because their training captions were too short (e.g., "blue shirt"). FG-CLIP uses extremely detailed descriptions of texture and material, making it much better for your specific use case.

# Industry-Specific Engines (Fashion & Textiles)


Because general-purpose models (even Jina) can sometimes struggle with technical textile terms (like "twill" vs. "sateen"), industry-specific players remain relevant:

- Ximilar: A Visual AI company that offers specialized "Fashion & Home Decor" search. They use "Deep Tagging" to extract specific attributes like pattern (striped, floral, checked) and material texture as part of the embedding process.

- Zhiyi Tech: A leader in apparel-industry search engines. They manage databases of hundreds of millions of clothing and fabric images and have proprietary models optimized specifically for pattern and fabric similarity.

# Summary

- Jina Embeddings v4 (Multi-Vector Mode): This is your best "out-of-the-box" choice. Use the multi-vector (late interaction) output if you need maximum precision on patterns.

- Llama-Embed-Nemotron-8B (NVIDIA): If you have the GPU budget (it's 8B params vs. Jina's 3B), this model currently holds the #1 Top-1 Precision spot on several retrieval leaderboards. It is excellent at following complex instructions (e.g., "Find a fabric with a diamond weave but a soft matte finish").

- ColPali-v1.2: While primarily known for "Document Search," the architecture of ColPali is essentially what Jina v4 perfected. It is the "gold standard" for searching visually rich items, making it very good at identifying patterns in images.

- GPT-4o is a VLM as well, so it should be possible to interrogate it but the embeddings aren't actually available AFAIK
