# Deepfake Video Detection – Attention-Based CNN (prototype)

Implements the architecture from the Second Review deck.

| Deck module | File |
|---|---|
| 1 Preprocessing (frame sampling, face crop, normalise) | `preprocess.py` |
| 2 Feature extraction (ResNeXt-50) | `models.py` |
| 3 Spatial self-attention  `A=softmax(QᵀK/√d)`, `F'=F+γ(V·Aᵀ)` | `models.py::SpatialSelfAttention` |
| 4 Temporal attention pooling  `α=softmax(wᵀtanh(WF'+b))` | `models.py::TemporalAttentionPooling` |
| 5 Classifier + cross-entropy + Adam | `models.py`, `train.py` |
| 6 Evaluation: within-domain, cross-dataset, compression (FFmpeg CRF) | `evaluate.py` |
| Attention map + IoU interpretability | `interpret.py`, `infer.py` |
| Identity-level split (no person leaks across splits) | `dataset.py` |
| Moderator demo UI | `app.py` |

`DeepfakeDetector(use_attention=False)` is the matched baseline (same backbone, plain frame averaging).

## Run
```bash
pip install -r requirements.txt
python smoke_test.py                                  # no data needed

# data layout: data_root/{real,fake}/<identity>/<video>.mp4
python train.py --data_root /data/ffpp --out runs/attn
python train.py --data_root /data/ffpp --out runs/base --baseline
python evaluate.py --ckpt runs/attn/best.pt --data_root /data/ffpp --split_json runs/attn/split.json --crf 28 35
python evaluate.py --ckpt runs/attn/best.pt --data_root /data/celebdf      # cross-dataset
python infer.py    --ckpt runs/attn/best.pt --video clip.mp4 --out result.png
python app.py      --ckpt runs/attn/best.pt                                # Gradio demo
```

## Notes
- γ is initialised to 0, so the attention model starts identical to the baseline and learns how much attention to mix in.
- Q/K are projected to 256-d to keep the 49×49 attention matrix cheap; V keeps 2048 channels for the residual add.
- Spatial map = mean attention each location receives, normalised to [0,1].
- Training uses a frozen backbone for the first epoch, then full fine-tuning.
- FaceForensics++ stores identities as video-ID pairs; arrange folders so the first sub-folder is the identity (or use the filename prefix).
