# Inspection Benchmark Scenario Matrix

Read this file to map a power inspection operator scenario to the benchmark dimensions that matter most.

## Voice-Agent Inspection

Use for operators such as RNN-T loss or softmax fusion that support noisy-site speech interaction.

Important dimensions:

- Batch size.
- Encoder frame length `T`.
- Target length `U`.
- Vocabulary size `V`.
- `fused_log_softmax` mode.
- Float dtype.
- Memory peak when available.

Evidence focus:

- Correctness against torchaudio or another named reference.
- Long-sequence memory pressure.
- Throughput or latency under fixed shape.
- Whether the result is training-only, inference-only, or both.

## 3D Reconstruction And Point-Cloud Preprocessing

Use for operators such as `unique_v3` that support point-cloud cleaning, sparse graph construction, or topology extraction.

Important dimensions:

- Element count or point count.
- Unique ratio.
- Dtype.
- Whether inverse and counts are enabled.
- Input ordering or distribution when relevant.
- CPU/GPU/NPU baseline scope.

Evidence focus:

- Exact output, inverse, and counts correctness.
- Small-shape overhead versus large-shape scaling.
- Preprocessing throughput for large inspection scenes.
- Clear separation between operator time and end-to-end reconstruction time.

## CV Defect Detection Pipeline Support

Use for future operators that support image preprocessing, postprocessing, detection heads, NMS, or feature transforms.

Important dimensions:

- Batch size.
- Image resolution or feature map size.
- Channel count.
- Dtype and layout.
- Number of boxes, anchors, or classes when applicable.
- End-to-end pipeline placement.

Evidence focus:

- Accuracy guardrail for detection-sensitive stages.
- Latency per image or per batch.
- Baseline implementation and synchronization.
- Whether timing includes host preprocessing or only the operator kernel.
