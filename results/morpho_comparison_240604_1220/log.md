### Wed, 4th July 12:20
Ran with this (label dependent)feature function for both models:

```
def get_features(word_with_pos: tuple[str, str], label: str) -> set[str]:
    """Label-dependent features"""
    form, pos = word_with_pos
    form = form.replace(" ", "_")
    features = set()
    
    for i in range(1, 6):
        if len(form) >= i:
            features.add(f"{pos} prefix={form[:i]} tag={label}")
            features.add(f"{pos} suffix={form[-i:]} tag={label}")
    
    return features
```

### RESULTS:
language,model,accuracy,time,n_train,n_labels
bul,Perceptron,0.747981338596806,200.64639043807983,44583,95
bul,MLR,0.7264489502960704,149.07205152511597,44583,95
rus,Perceptron,0.6536874214965009,218.7036099433899,44583,101
rus,MLR,0.6610443208325857,162.77459025382996,44583,101