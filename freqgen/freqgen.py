from collections import Counter, Iterable, defaultdict
from itertools import chain, islice, product
from warnings import warn

import Bio.Data.CodonTable
import numpy as np
from Bio.Seq import Seq

# create the genetic_codes and codons_for_aa dicts
genetic_codes = {}
codons_for_aa = {}
synonymous_codons = {}

for code_id, genetic_code in Bio.Data.CodonTable.unambiguous_dna_by_id.items():
    # create genetic_codes dict
    # Ex: {'TTT': 'F', 'TTC': 'F', 'TTA': 'L'...
    table = genetic_code.forward_table
    for codon in genetic_code.stop_codons:
        table[codon] = "*"
    genetic_codes[code_id] = table

    # create codons_for_aa dict
    # Ex: defaultdict(<class 'list'>, {'F': ['TTT', 'TTC']...
    _codons_for_aa = defaultdict(list)
    for key, value in table.items():
        _codons_for_aa[value].append(key)
    codons_for_aa[code_id] = dict(_codons_for_aa)

    # create a list of synonymous_codons for each codon, including the original codon
    # Ex: {'TTT': ['TTT', 'TTC']...
    synonymous_codons[code_id] = {
        codon: codons_for_aa[code_id][genetic_codes[code_id][codon]]
        for codon in genetic_codes[code_id].keys()
    }


def amino_acid_seq(length, frequencies):
    """Generates an amino acid sequence given frequencies of each amino acid.

    Args:
        length (int): The length of the amino acid sequence to generate.
        frequencies (dict): A dictionary containing a mapping of each amino acid to its frequency.

    Note:
        The sum of all the values in ``frequencies`` must be 1.

    Returns:
        str: An amino acid sequence with the given frequencies.

    Raises:
        ValueError: When the length of the sequence is invalid or when the probabilities do not sum to 1.

    Example:
        >>> from Bio import SeqIO
        >>> seq = SeqIO.read("beta_lactamase.fasta", "fasta").seq
        >>> frequencies = k_mer_frequencies(seq, 1)
        >>> amino_acid_seq(25, frequencies)
    """

    if length <= 0:
        raise ValueError("Length must be a positive integer")

    sequence = ""
    amino_acids, frequencies = zip(*frequencies[list(frequencies.keys())[0]].items())
    for i in range(length):
        sequence += np.random.choice(amino_acids, p=frequencies)
    return sequence


def amino_acids_to_codons(aa_seq, codon_frequencies, genetic_code=11):
    """Generates a DNA representation of an amino acid sequence.

    Args:
        aa_seq (str): The amino acids to convert to DNA.
        codon_frequencies (dict): A dictionary of codon frequencies for each amino acid. For each amino acid, the sum of the frequencies of its codons must be 1.
        genetic_code (int, optional): The genetic code to use when converting to DNA. Defaults to 11, the standard genetic code.

    Returns:
        str: A DNA sequence with the given codon usage.

    Example:
        >>> from Bio import SeqIO
        >>> seq = SeqIO.read("sequence.fasta", "fasta").seq
        >>> frequencies = codon_frequencies(seq)
        >>> amino_acids_to_codons("INQTEL", frequencies)
        'ATAAATCAAACCGAACTT'
    """

    codons_dict = codons_for_aa[genetic_code]

    # generate the sequence
    sequence = []
    for aa in aa_seq:
        try:
            codons = codons_dict[aa]
            sequence.append(
                np.random.choice(
                    codons, p=[codon_frequencies[codon] for codon in codons]
                )
            )
        except KeyError:
            pass

    if len(sequence) != len(aa_seq):
        raise KeyError("Missing codon frequency.")

    return "".join(sequence)


def codon_frequencies(seq, mode="absolute", genetic_code=11):
    """Calculates the frequency of each codon

    Absolute mode is such that the total of the dictionary's values is equal to one.
    Relative mode is such that the sum of each amino acid's codons' frequencies is equal to one.

    Args:
        seq (str or list): The DNA sequence(s).
        mode (str, optional): One of "absolute" or "relative". Defaults to "absolute"
        genetic_code (int, optional): The genetic code to use when converting to DNA. Defaults to 11, the standard genetic code.

    Returns:
        dict: The codon frequencies of each codon.

    Raises:
        ValueError: When the sequence length is not divisible into codons, i.e. when sequence length is not divisible by three.

    Example:
        >>> from Bio import SeqIO
        >>> seq = SeqIO.read("sequence.fasta", "fasta").seq
        >>> codon_frequencies(seq)
        {'AAA': 0.016129032258064516,
         'AAC': 0.016129032258064516,
         'AAG': 0.016129032258064516,
         'AAT': 0.016129032258064516,
         'ACA': 0,
         'ACC': 0.016129032258064516,
         'ACG': 0,
         'ACT': 0.016129032258064516,
         'AGA': 0,
         'AGC': 0,
         'AGG': 0,
         'AGT': 0,
         'ATA': 0.016129032258064516,
         'ATC': 0,
         'ATG': 0.03225806451612903,
         'ATT': 0,
         'CAA': 0.04838709677419355,
         'CAC': 0,
         'CAG': 0,
         'CAT': 0.03225806451612903,
         'CCA': 0.03225806451612903,
         'CCC': 0.016129032258064516,
         'CCG': 0.04838709677419355,
         'CCT': 0.03225806451612903,
         'CGA': 0.016129032258064516,
         'CGC': 0,
         'CGG': 0.03225806451612903,
         'CGT': 0,
         'CTA': 0,
         'CTC': 0,
         'CTG': 0.03225806451612903,
         'CTT': 0.0967741935483871,
         'GAA': 0.016129032258064516,
         'GAC': 0,
         'GAG': 0,
         'GAT': 0.016129032258064516,
         'GCA': 0,
         'GCC': 0.06451612903225806,
         'GCG': 0.016129032258064516,
         'GCT': 0,
         'GGA': 0.03225806451612903,
         'GGC': 0.016129032258064516,
         'GGG': 0.016129032258064516,
         'GGT': 0,
         'GTA': 0,
         'GTC': 0.016129032258064516,
         'GTG': 0,
         'GTT': 0.04838709677419355,
         'TAA': 0,
         'TAC': 0,
         'TAG': 0,
         'TAT': 0.03225806451612903,
         'TCA': 0.016129032258064516,
         'TCC': 0.03225806451612903,
         'TCG': 0,
         'TCT': 0.03225806451612903,
         'TGA': 0,
         'TGC': 0.016129032258064516,
         'TGG': 0.016129032258064516,
         'TGT': 0,
         'TTA': 0,
         'TTC': 0.04838709677419355,
         'TTG': 0,
         'TTT': 0.03225806451612903}
    """

    if isinstance(seq, (list, tuple)):
        for _seq in seq:
            if (
                len(_seq) % 3 != 0
            ):  # check to ensure sequence contains only complete codons
                raise ValueError("Sequence length must be divisible by 3.")
        seq = "".join([str(seq) for seq in seq])

    if len(seq) % 3 != 0:  # check to ensure sequence contains only complete codons
        raise ValueError("Sequence length must be divisible by 3.")
    seq = str(seq).upper()

    seq = [
        seq[i : i + 3] for i in range(0, len(seq), 3)
    ]  # slices the sequence into individual codons
    codon_count = Counter(seq)
    frequencies = {
        key: (float(value) / len(seq)) for (key, value) in codon_count.items()
    }

    # collections.Counter returns a dictionary with counts of all the codons
    # present. To ensure a 64-D vector, we make sure all codons are present in
    # the dictionary.
    for codon in genetic_codes[genetic_code]:
        try:
            frequencies[codon]
        except KeyError:
            frequencies[codon] = 0
    if mode == "absolute":
        return frequencies

    if mode == "relative":
        relative = {}
        for i in synonymous_codons[genetic_code].keys():
            try:
                relative[i] = frequencies[i] / sum(
                    (frequencies[codon] for codon in synonymous_codons[genetic_code][i])
                )  # divide the occurence of a codon by the total number of its synonyms
            except ZeroDivisionError:
                relative[i] = 1 / len(
                    synonymous_codons[genetic_code][i]
                )  # if an amino acid is never used in the reference set, then all its codons are used equally
        return relative

    else:
        raise ValueError("Mode must be either absolute or relative.")


def k_mers(seq, k):
    """Yields all *k*-mers in the input sequence with repeats.

    Args:
        seq (str): The sequence for which to generate *k*-mers.
        k (int): the length of the *k*-mers.

    Yields:
        str: the next *k*-mer

    Raises:
        ValueError: When the value of *k* is less than the length of the sequence, k <= 0, or len(seq) is 0.

    Example:
        >>> list(k_mers("GATTACA", 1))
        ['G', 'A', 'T', 'T', 'A', 'C', 'A']
        >>> list(k_mers("GATTACA", 2))
        ['GA', 'AT', 'TT', 'TA', 'AC', 'CA']
        >>> list(k_mers("GATTACA", 3))
        ['GAT', 'ATT', 'TTA', 'TAC', 'ACA']
        >>> list(k_mers("GATTACA", 4))
        ['GATT', 'ATTA', 'TTAC', 'TACA']
        >>> k_mers("GATTACA", 4)
        <generator object k_mers at 0x10831d258>
    """

    # error checking
    if k > len(seq):
        raise ValueError(
            "k (%i) may not be less then length of seq (%i)." % (k, len(seq))
        )
    elif not seq:
        raise ValueError("seq length may not be zero")
    elif k <= 0:
        raise ValueError("k may not be <= zero")

    it = iter(seq)
    result = tuple(islice(it, k))
    if len(result) == k:
        yield "".join(result)
    for elem in it:
        result = result[1:] + (elem,)
        yield "".join(result)


def k_mer_frequencies(
    seq, k, include_missing=True, vector=False, codons=False, genetic_code=11
):
    """Calculates relative frequencies of each *k*-mer in the sequence.

    Args:
        seq (str or list): The sequence(s) to for which to generate *k*-mer frequencies.
        k (int or list): the length of the *k*-mer(s).
        include_missing (bool, optional): If True, include missing *k*-mers as having a frequency of 0. Only supports DNA *k*-mers. Defaults to False.
        vector (bool, optional): Return a 1-D Numpy array of the *k*-mer frequencies, ordered by *k*-mers alphabetically. If True, ``include_missing`` must also be True. Defaults to False.
        codons (bool, optional): Whether to include a codon usage entry in the resulting dictionary. Defaults to False.
        genetic_code (int, optional): The genetic code to use when converting to DNA. Defaults to 11, the standard genetic code.

    Returns:
        dict: A dict in which the keys are *k* values and the values are dictionaries mapping *k*-mers to floats of their frequencies.

    Raises:
        ValueError: When an invalid value of k is provided or ``include_missing`` is False and ``vector`` is True.
        ValueError: When ``codons`` and ``vector`` are both True.
        ValueError: When ``k`` or ``seq`` is not provided.

    Example:
        >>> k_mer_frequencies("INQTEL", 1, include_missing=False)
        {1: {'E': 0.16666666666666666,
             'I': 0.16666666666666666,
             'L': 0.16666666666666666,
             'N': 0.16666666666666666,
             'Q': 0.16666666666666666,
             'T': 0.16666666666666666}}

        >>> k_mer_frequencies("GATGATGGC", [1, 2], include_missing=False)
        {1: {'A': 0.2222222222222222,
             'C': 0.1111111111111111,
             'G': 0.4444444444444444,
             'T': 0.2222222222222222},
         2: {'AT': 0.25, 'GA': 0.25, 'GC': 0.125, 'GG': 0.125, 'TG': 0.25}}

        >>> k_mer_frequencies(["A", "T"], 1, include_missing=False)
        {1: {'A': 0.5, 'T': 0.5}}

        >>> k_mer_frequencies("GATGATGGC", 2, include_missing=True)
        {2: {'AA': 0,
             'AC': 0,
             'AG': 0,
             'AT': 0.25,
             'CA': 0,
             'CC': 0,
             'CG': 0,
             'CT': 0,
             'GA': 0.25,
             'GC': 0.125,
             'GG': 0.125,
             'GT': 0,
             'TA': 0,
             'TC': 0,
             'TG': 0.25,
             'TT': 0}}

        >>> k_mer_frequencies("GATGATGGC", 2, include_missing=True, vector=True)
        array([0.   , 0.   , 0.   , 0.25 , 0.   , 0.   , 0.   , 0.   , 0.25 ,
               0.125, 0.125, 0.   , 0.   , 0.   , 0.25 , 0.   ])
    """

    if not include_missing and vector:
        raise ValueError("May not create vector without including missing kmers.")
    elif not k:  # for when k == 0 or []
        raise ValueError("Must provide a value for k")
    elif not seq:  # for when seq == ""
        raise ValueError("Must provide seq(s)")
    elif codons and vector:
        raise ValueError("Cannot vectorize codons.")

    # ensure there is a list of k values, even if it only has one element
    if not isinstance(k, Iterable):
        k = [k]
    else:
        k = sorted(k)

    output = {}

    # ditto for sequence(s)
    if isinstance(seq, (str, bytes, Seq)):
        seq = [seq]

    for _k in k:

        # check the value of k
        if _k < 1:
            raise ValueError("Invalid value of k. May not be less than 1.")

        # get all the k-mers for the seqs
        _seqs = []
        for _seq in [list(k_mers(_seq.upper(), _k)) for _seq in seq]:
            _seqs.extend(_seq)

        # determine their frequencies
        count = Counter(_seqs)
        total_k_mer_count = sum(count.values())
        frequencies = {
            k_mer: value / total_k_mer_count for k_mer, value in count.items()
        }

        if include_missing:
            defaults = {"".join(x): 0 for x in product("ATGC", repeat=_k)}
            frequencies = {**defaults, **frequencies}
        if vector:
            frequencies = sorted(list(frequencies.items()), key=lambda x: x[0])
            frequencies = np.fromiter(
                (x[1] for x in frequencies), float, count=len(frequencies)
            )
        output[_k] = frequencies

    if vector:
        return np.array(list(chain.from_iterable([output[_k] for _k in k])))

    # syntactic sugar to make capturing codon usage easier
    if codons:
        output["codons"] = codon_frequencies(seq, genetic_code=genetic_code)

    return output
