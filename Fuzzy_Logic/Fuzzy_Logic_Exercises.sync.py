# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fuzzy logic exercises
#
# In this notebook, you will write code to compute the cardinality of finite fuzzy sets and basic fuzzy logic opreators: union, intersection, and complement. For this exercise, you should not use any libraries, but rather use only basic Python.

# %% [markdown]
# Let's define some fuzzy sets:

# %%
# Universal set
U = ("a", "b", "c", "d", "e", "f", "g")

# A and B are finite fuzzy sets:
A = {"a": 0.4, "b": 1, "c": 0.6, "d": 0.0, "e": 0.5, "f": 0.3, "g": 0.9}
B = {"a": 0.6, "b": 0.2, "c": 0.6, "d": 0.7, "e": 0.99, "f": 0.5, "g": 0}

# %% [markdown]
# ## Cardinality and complement

# %% [markdown]
# Implement code that can outputs the cardinality of $A$ and the cardinality of $B$:

# %%
# implement your code to compute the cardinality of A and B here:
print(sum(A.values()))
print(sum(B.values()))

# %% [markdown]
# The cardinality of $A$ should be 3.7
#
# The cardinality of $B$ should be 3.59

# %% [markdown]
# Now, implement code that computes C as the complement of $A$, that is $C = \bar{A}$:

# %%
# implement your code to compute the complement of A here:
C = dict()
for key, val in A.items():
    C[key] = 1 - val
print(C)

# %% [markdown]
# ## Union and intersection

# %% [markdown]
# Implement code that computes $I$ as the intersection of $A$ and $B$, that is $I = A \cap B$:

# %%
# implement your code to compute the intersection between A and B here:
D = dict()
for keyA, valueA in A.items():
    for keyB, valueB in B.items():
        if keyA == keyB:
            D[keyA] = min(valueA, valueB)
print(D)

# %% [markdown]
# Finally, write code that computes $N$ as the union of $A$ and $B$, that is $N = A \cup B$

# %%
# implement your code to compute the union of A and B here:
E = dict()
for keyA, valueA in A.items():
    for keyB, valueB in B.items():
        if keyA == keyB:
            E[keyA] = max(valueA, valueB)
print(E)

# %% [markdown]
# Great! You reached the end of the exercises in this notebook. However, note that the membership degree was specified for all elements of $U$ in both the dicts ``A`` and ``B``, even for elements with member degree 0, for instance ``d``in ``A`` (``A = { ... 'd': 0.0, ...  }``. If you haven't already done so, try to make your code robust to fuzzy set declarations as dicts that only contain a key-pairs for the elements with a positive membership degree. Hence, make the code work even if you remove ``d`` from the ``A`` and ``g`` from ``B`` in the declaration of the dicts at the top of this notebook.
