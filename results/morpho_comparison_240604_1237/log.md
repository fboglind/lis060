### Wed, 4th July 12:37
Ran with this (label dependent for Perceptron, Label Agnostic for MLR)feature function for both models:

```def get_features(word_with_pos: tuple[str, str], label: str | None = None) -> set[str]:
    """Extract features from word form and POS.
    If label is provided, creates label-dependent features (for Perceptron).
    If label is None, creates label-agnostic features (for MLR).
    """
    form, pos = word_with_pos
    form = form.replace(" ", "_")  # Handle multi-word expressions
    features = set()
    
    if label is None:
        # Label-agnostic features for MLR
        for i in range(1, 6):
            if len(form) >= i:
                features.add(f"{pos} prefix={form[:i]}")
                features.add(f"{pos} suffix={form[-i:]}")
    else:
        # Label-dependent features for Perceptron
        for i in range(1, 6):
            if len(form) >= i:
                features.add(f"{pos} prefix={form[:i]} tag={label}")
                features.add(f"{pos} suffix={form[-i:]} tag={label}")
    
    return features
```
### RESULTS:
language,model,accuracy,time,n_train,n_labels
bul,Perceptron,0.7535438722411627,197.7259156703949,44583,95
bul,MLR,0.7264489502960704,137.10823583602905,44583,95
rus,Perceptron,0.6553023506190562,214.49429988861084,44583,101
rus,MLR,0.6610443208325857,160.72304725646973,44583,101
