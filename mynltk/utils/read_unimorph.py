# def read_unimorph(
#     filepath: str, testing=False, use_surface_form=False
# ) -> list:
#     """Read UniMorph data and prepare for feature-based analysis
#     Args:
#         filepath: Path to datafile (str)
#         testing: Use this for the test set (bool)
#         use_surface_form: False allows for multiword expressions (bool)
#     Returns:
#         data: list of words and tags (list[tuple])
#     """
#     data = []
#     with open(filepath, "r", encoding="utf-8") as f:
#         for line in f:
#             base, inflected, tags = line.strip().split("\t")
#             # Handle multi-word expressions
#             inflected = (
#                 inflected.replace(" ", "_") if use_surface_form else inflected
#             )
#             pos = tags.split(";")[0]
#             if testing:
#                 data.append((inflected, pos))  # Just the input
#             else:
#                 data.append(
#                     ((inflected, pos), tags)
#                 )  # ((form, POS), full tag)
#     return data

def read_unimorph(filepath: str, testing=False, use_surface_form=False) -> list:
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):  # Skip empty lines and comments
                continue
            
            parts = line.split("\t")
            if len(parts) != 3:  # Skip malformed lines
                print(f"Warning: Skipping malformed line {line_num}: {line}")
                continue
                
            base, inflected, tags = parts
            #Handle multi-word expressions
            inflected = (
                inflected.replace(" ", "_") if use_surface_form else inflected
            )
            pos = tags.split(";")[0]
            if testing:
                data.append((inflected, pos))  # Just the input
            else:
                data.append(
                    ((inflected, pos), tags)
                )  # ((form, POS), full tag)
    return data