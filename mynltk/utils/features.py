def get_features(word_with_pos: tuple[str, str], label: str | None = None) -> set[str]:
    form, pos = word_with_pos
    form = form.replace(" ", "_")  # Handle multi-word expressions
    features = set()

    # Optional label part
    label_suffix = f" tag={label}" if label else ""

    for i in range(1, 6):
        if len(form) >= i:
            prefix = form[:i]
            suffix = form[-i:]
            features.add(f"{pos} prefix={prefix}{label_suffix}")
            features.add(f"{pos} suffix={suffix}{label_suffix}")

    return features

#CLAUDE:
# def get_features(word_with_pos: tuple[str, str], label: str | None = None) -> set[str]:
#     form, pos = word_with_pos
#     features = set()
    
#     # Label-agnostic features only
#     for i in range(1, 6):
#         if len(form) >= i:
#             features.add(f"{pos} prefix={form[:i]}")
#             features.add(f"{pos} suffix={form[-i:]}")
    
#     return features


# def get_features(word_with_pos: tuple, label: str) -> set[str]:
#     form, pos = word_with_pos
#     form = form.replace(" ", "_")
#     features = set()

#     for i in range(1, 6):
#         if len(form) >= i:
#             prefix = form[:i]
#             suffix = form[-i:]
#             features.add(f"{pos} prefix={prefix} tag={label}")
#             features.add(f"{pos} suffix={suffix} tag={label}")

#     return features

# def get_features(word_with_pos: tuple, label: str) -> set[str]:
#         """Extract prefix/suffix features from the word form and POS
#         Args:
#             word_with_pos: A tuple containing the word form and its POS
#             label: The morphological label
#             Returns:
#                 A set of features for the word form and POS
#         """
#         form, pos = word_with_pos
#         form = form.replace(" ", "_")  # Handle multi-word expressions
#         features = set()

#         # Generate prefixes
#         for i in range(1, 6):
#             if len(form) >= i:
#                 prefix = form[:i]
#                 features.add(f"{pos} prefix={prefix} tag={label}")

#         # Generate suffixes
#         for i in range(1, 6):
#             if len(form) >= i:
#                 suffix = form[-i:]
#                 features.add(f"{pos} suffix={suffix} tag={label}")

#         return features