import numpy as np
def predict_crop(model_cnn, rf_model, images, seqs, ids):
    # images: CHW stacked array saved earlier -> convert to HWC for CNN
    # images shape (N, C, H, W)
    imgs_hwc = np.transpose(images, (0,2,3,1))
    embeddings = model_cnn.predict(imgs_hwc)
    # Flatten seqs into fixed-length features (if variable length, simple aggregation)
    seq_feats = np.array([s.mean(axis=0) for s in seqs])
    X_tab = np.hstack([seq_feats, embeddings])
    preds = rf_model.predict(X_tab)
    probs = rf_model.predict_proba(X_tab) if hasattr(rf_model, 'predict_proba') else None
    out = []
    for i, cid in enumerate(ids):
        out.append({
            'cell_id': cid,
            'label': int(preds[i]),
            'confidence': float(probs[i].max()) if probs is not None else None
        })
    return out
